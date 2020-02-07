from typing import Iterator, List, Set
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
from dataclasses import dataclass
from cjwmodule.util.colnames import gen_unique_clean_colnames

def wide_to_long(table: pd.DataFrame, colname: str) -> pd.DataFrame:
    # Check all values are the same type
    value_table = table[set(table.columns).difference([colname])]

    if value_table.empty:
        # Avoids 'No objects to concatenate' when colname is categorical and
        # there are no values or other columns
        return pd.DataFrame({colname: [], "variable": [], "value": []}, dtype=str)

    value_dtypes = value_table.dtypes
    are_numeric = value_dtypes.map(is_numeric_dtype)
    are_datetime = value_dtypes.map(is_datetime64_dtype)
    are_text = ~are_numeric & ~are_datetime
    if not are_numeric.all() and not are_datetime.all() and not are_text.all():
        # Convert mixed values so they're all text. Values must all be the same
        # type.
        to_convert = value_table.columns[~are_text]
        na = table[to_convert].isna()
        table.loc[:, to_convert] = table[to_convert].astype(str)
        table.loc[:, to_convert][na] = np.nan

        cols_str = ", ".join(f'"{c}"' for c in to_convert)
        error = (
            f"Columns {cols_str} were auto-converted to Text because the "
            "value column cannot have multiple types."
        )
        quick_fixes = [
            {
                "text": f"Convert {cols_str} to text",
                "action": "prependModule",
                "args": ["converttotext", {"colnames": list(to_convert)}],
            }
        ]
    else:
        error = ""
        quick_fixes = []

    table = pd.melt(table, id_vars=[colname])
    table.sort_values(colname, inplace=True)
    table.reset_index(drop=True, inplace=True)

    if error:
        return {"dataframe": table, "error": error, "quick_fixes": quick_fixes}
    else:
        return table


def long_to_wide(
    table: pd.DataFrame, keycolnames: List[str], varcolname: str
) -> pd.DataFrame:
    warnings = []
    quick_fixes = []

    varcol = table[varcolname]
    if varcol.dtype != object and not hasattr(varcol, "cat"):
        # Convert to str, in-place
        warnings.append(
            (
                'Column "%s" was auto-converted to Text because column names '
                "must be text."
            )
            % varcolname
        )
        quick_fixes.append(
            {
                "text": 'Convert "%s" to text' % varcolname,
                "action": "prependModule",
                "args": ["converttotext", {"colnames": [varcolname]}],
            }
        )
        na = varcol.isnull()
        varcol = varcol.astype(str)
        varcol[na] = np.nan
        table[varcolname] = varcol

    # Remove empty values, in-place. Empty column headers aren't allowed.
    # https://www.pivotaltracker.com/story/show/162648330
    empty = varcol.isin([np.nan, pd.NaT, None, ""])
    n_empty = np.count_nonzero(empty)
    if n_empty:
        if n_empty == 1:
            text_empty = "1 input row"
        else:
            text_empty = "{:,d} input rows".format(n_empty)
        warnings.append('%s with empty "%s" were removed.' % (text_empty, varcolname))
        table = table[~empty]
        table.reset_index(drop=True, inplace=True)

    table.set_index(keycolnames + [varcolname], inplace=True, drop=True)
    if np.any(table.index.duplicated()):
        return "Cannot reshape: some variables are repeated"
    if len(table.columns) == 0:
        return (
            "There is no Value column. "
            "All but one table column must be a Row or Column variable."
        )
    if len(table.columns) > 1:
        return (
            "There are too many Value columns. "
            "All but one table column must be a Row or Column variable. "
            "Please drop extra columns before reshaping."
        )

    table = table.unstack()
    table.columns = [col[-1] for col in table.columns.values]
    table.reset_index(inplace=True)

    if warnings:
        return {
            "dataframe": table,
            "error": "\n".join(warnings),
            "quick_fixes": quick_fixes,
        }
    else:
        return table


def render(table, params, *, input_columns):
    dir = params["direction"]
    colname = params["colnames"]  # bad param name! It's single-column
    varcol = params["varcol"]

    # no columns selected and not transpose, NOP
    if not colname and dir != "transpose":
        return table

    if dir == "widetolong":
        return wide_to_long(table, colname)

    elif dir == "longtowide":
        if not varcol:
            # gotta have this parameter
            return table

        keys = [colname]

        has_second_key = params["has_second_key"]
        # If second key is used and present, append it to the list of columns
        if has_second_key:
            second_key = params["second_key"]
            if second_key in table.columns:
                keys.append(second_key)

        if varcol in keys:
            return "Cannot reshape: column and row variables must be different"

        return long_to_wide(table, keys, varcol)

    elif dir == "transpose":
        return transpose(
            table,
            # Backwards-compat because we published it like this way back when
            {"firstcolname": "New Column"},
            input_columns=input_columns,
        )


def _migrate_params_v0_to_v1(params):
    # v0: menu item indices. v1: menu item labels
    v1_dir_items = ["widetolong", "longtowide", "transpose"]
    params["direction"] = v1_dir_items[params["direction"]]
    return params


def migrate_params(params):
    # Convert numeric direction parameter to string labels, if needed
    if isinstance(params["direction"], int):
        params = _migrate_params_v0_to_v1(params)

    return params


# COPY/PASTE from the `transpose` module.
# EDIT WITH THESE STEPS ONLY:
# 1. Find the bug in the `transpose` module. Unit-test it; fix; deploy.
# 2. Copy/paste the `transpose` module's "render()" method here.
# 3. Rename `render(...)` to `transpose(table, params)`
# hard-code settings for now. TODO have Workbench pass render(..., settings=...)
@dataclass
class Settings:
    MAX_COLUMNS_PER_TABLE: int
    MAX_BYTES_PER_COLUMN_NAME: int


settings = Settings(99, 120)


@dataclass
class GenColnamesResult:
    names: List[str]
    """All column names for the output table (even the first column)."""

    warnings: List[str]
    """All the things we should tell the user about how we tweaked names."""


def _gen_colnames_and_warn(
    first_colname: str, first_column: pd.Series
) -> GenColnamesResult:
    """
    Generate transposed-table column names.

    If `first_colname` is empty, `column.name` is the first output column. If
    both are empty, auto-generate the column name (and warn).

    Warn if ASCII-cleaning names, renaming duplicates, truncating names or
    auto-generating names.

    Assume `first_column` is text without nulls.
    """
    n_ascii_cleaned = 0
    first_ascii_cleaned = None
    n_default = 0
    first_default = None
    n_truncated = 0
    first_truncated = None
    n_numbered = 0
    first_numbered = None

    input_names = [first_colname or first_column.name]
    input_names.extend(list(first_column.values))

    names = []

    for uccolname in gen_unique_clean_colnames(input_names, settings=settings):
        name = uccolname.name
        names.append(name)
        if uccolname.is_ascii_cleaned:
            if n_ascii_cleaned == 0:
                first_ascii_cleaned = name
            n_ascii_cleaned += 1
        if uccolname.is_default:
            if n_default == 0:
                first_default = name
            n_default += 1
        if uccolname.is_truncated:
            if n_truncated == 0:
                first_truncated = name
            n_truncated += 1
        if uccolname.is_numbered:
            if n_numbered == 0:
                first_numbered = name
            n_numbered += 1

    warnings = []
    if n_ascii_cleaned > 0:
        warnings.append(
            "Removed special characters from %d column names (see “%s”)"
            % (n_ascii_cleaned, first_ascii_cleaned)
        )
    if n_default > 0:
        warnings.append(
            "Renamed %d column names (because values were empty; see “%s”)"
            % (n_default, first_default)
        )
    if n_truncated > 0:
        warnings.append(
            "Truncated %d column names (to %d bytes each; see “%s”)"
            % (n_truncated, settings.MAX_BYTES_PER_COLUMN_NAME, first_truncated)
        )
    if n_numbered > 0:
        warnings.append(
            "Renamed %d duplicate column names (see “%s”)"
            % (n_numbered, first_numbered)
        )

    return GenColnamesResult(names, warnings)


def transpose(table, params, *, input_columns):
    warnings = []
    colnames_auto_converted_to_text = []

    if len(table) > settings.MAX_COLUMNS_PER_TABLE:
        table = table.truncate(after=settings.MAX_COLUMNS_PER_TABLE - 1)
        warnings.append(
            f"We truncated the input to {settings.MAX_COLUMNS_PER_TABLE} rows so the "
            "transposed table would have a reasonable number of columns."
        )

    if not len(table.columns):
        # happens if we're the first module in the module stack
        return pd.DataFrame()

    column = table.columns[0]
    first_column = table[column]
    table.drop(column, axis=1, inplace=True)

    if input_columns[column].type != "text":
        warnings.append(f'Headers in column "A" were auto-converted to text.')
        colnames_auto_converted_to_text.append(column)

    # Ensure headers are string. (They will become column names.)
    # * categorical => str
    # * nan => ""
    # * non-text => str
    na = first_column.isna()
    first_column = first_column.astype(str)
    first_column[na] = ""  # Empty values are all equivalent

    gen_headers_result = _gen_colnames_and_warn(params["firstcolname"], first_column)
    warnings.extend(gen_headers_result.warnings)

    input_types = set(c.type for c in input_columns.values() if c.name != column)
    if len(input_types) > 1:
        # Convert everything to text before converting. (All values must have
        # the same type.)
        to_convert = [c for c in table.columns if input_columns[c].type != "text"]
        colnames_auto_converted_to_text.extend(to_convert)
        if len(to_convert) == 1:
            start = f'Column "{to_convert[0]}" was'
        else:
            colnames = ", ".join(f'"{c}"' for c in to_convert)
            start = f"Columns {colnames} were"
        warnings.append(
            f"{start} auto-converted to Text because all columns must have "
            "the same type."
        )

        for colname in to_convert:
            # TODO respect column formats ... and nix the quick-fix?
            na = table[colname].isnull()
            table[colname] = table[colname].astype(str)
            table[colname][na] = np.nan

    # The actual transpose
    table.index = gen_headers_result.names[1:]
    ret = table.T
    # Set the name of the index: it will become the name of the first column.
    ret.index.name = gen_headers_result.names[0]
    # Make the index (former colnames) a column
    ret.reset_index(inplace=True)

    if warnings and colnames_auto_converted_to_text:
        colnames = ", ".join(f'"{c}"' for c in colnames_auto_converted_to_text)
        return {
            "dataframe": ret,
            "error": "\n".join(warnings),
            "quick_fixes": [
                {
                    "text": f"Convert {colnames} to text",
                    "action": "prependModule",
                    "args": [
                        "converttotext",
                        {"colnames": colnames_auto_converted_to_text},
                    ],
                }
            ],
        }
    if warnings:
        return (ret, "\n".join(warnings))
    else:
        return ret
# END copy/paste
