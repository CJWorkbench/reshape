from typing import Iterator, List, Set
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
from cjwmodule import i18n
from dataclasses import dataclass
from cjwmodule.util.colnames import gen_unique_clean_colnames_and_warn


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
        error = {
            "message": i18n.trans(
                "wide_to_long.badColumns.mixedTypes.error",
                "{n_columns, plural, other{Columns {column_names} were} one {Column {column_names} was}}"
                "auto-converted to Text because the "
                "value column cannot have multiple types.",
                {
                    "n_columns": len(to_convert),
                    "column_names": cols_str
                }
            ),
            "quickFixes": [
                {
                    "text": i18n.trans(
                        "wide_to_long.badColumns.mixedTypes.quick_fix.text",
                        "Convert {cols_str} to text", 
                        {"column_names": cols_str}
                    ),
                    "action": "prependModule",
                    "args": ["converttotext", {"colnames": list(to_convert)}],
                }
            ]
        }
    else:
        error = None

    table = pd.melt(table, id_vars=[colname])
    table.sort_values(colname, inplace=True)
    table.reset_index(drop=True, inplace=True)

    if error:
        return (table, error)
    else:
        return table


def long_to_wide(
    table: pd.DataFrame, keycolnames: List[str], varcolname: str
) -> pd.DataFrame:
    warnings = []

    varcol = table[varcolname]
    if varcol.dtype != object and not hasattr(varcol, "cat"):
        # Convert to str, in-place
        warnings.append(
            {
                "message": i18n.trans(
                    "long_to_wide.badColumn.notText.error",
                    'Column "{column_name}" was auto-converted to Text '
                    "because column names must be text.",
                    {"column_name": varcolname}
                ),
                "quickFixes": [{
                    "text": i18n.trans(
                        "long_to_wide.badColumn.notText.quick_fix.text",
                        'Convert "{column_name}" to text',
                        {"column_name": varcolname},
                    ),
                    "action": "prependModule",
                    "args": ["converttotext", {"colnames": [varcolname]}],
                }]
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
        warnings.append(i18n.trans(
            "long_to_wide.badRows.emptyColumnHeaders.warning",
            '{n_rows, plural, '
            '  one {# row with empty "{column_name}" was removed.}'
            '  other {# rows with empty "{column_name}" were removed.}'
            '}',
            {"n_rows": n_empty, "column_name": varcolname}
        ))
        table = table[~empty]
        table.reset_index(drop=True, inplace=True)

    table.set_index(keycolnames + [varcolname], inplace=True, drop=True)
    if np.any(table.index.duplicated()):
        return i18n.trans(
            "long_to_wide.error.repeatedVariables", 
            "Cannot reshape: some variables are repeated"
        )
    if len(table.columns) == 0:
        return i18n.trans(
            "long_to_wide.error.noValueColumn", 
            "There is no Value column. "
            "All but one table column must be a Row or Column variable."
        )
    if len(table.columns) > 1:
        return i18n.trans(
            "long_to_wide.error.tooManyValueColumns", 
            "There are too many Value columns. "
            "All but one table column must be a Row or Column variable. "
            "Please drop extra columns before reshaping."
        )

    table = table.unstack()
    table.columns = [col[-1] for col in table.columns.values]
    table.reset_index(inplace=True)

    if warnings:
        return (table, warnings)
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
            return i18n.trans(
                "error.sameColumnAndRowVariables", 
                "Cannot reshape: column and row variables must be different"
            )

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
# 4. Add `"transpose."` as a prefix to `i18n.trans` ids
# 5. Copy translations of transpose messages (adapting their ids as in the previous step)
#    to `locale/{locale}/messages.po for all locales except `"en"`

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
    input_names = [first_colname or first_column.name]
    input_names.extend(list(first_column.values))

    names, warnings = gen_unique_clean_colnames_and_warn(input_names, settings=settings)

    return GenColnamesResult(names, warnings)


def transpose(table, params, *, input_columns):
    warnings = []
    colnames_auto_converted_to_text = []

    if len(table) > settings.MAX_COLUMNS_PER_TABLE:
        table = table.truncate(after=settings.MAX_COLUMNS_PER_TABLE - 1)
        warnings.append(
            i18n.trans(
                "transpose.warnings.tooManyRows",
                "We truncated the input to {max_columns} rows so the "
                "transposed table would have a reasonable number of columns.",
                {"max_columns": settings.MAX_COLUMNS_PER_TABLE},
            )
        )

    if not len(table.columns):
        # happens if we're the first module in the module stack
        return pd.DataFrame()

    column = table.columns[0]
    first_column = table[column]
    table.drop(column, axis=1, inplace=True)

    if input_columns[column].type != "text":
        warnings.append(
            {
                "message": i18n.trans(
                    "transpose.warnings.headersConvertedToText.message",
                    'Headers in column "{column_name}" were auto-converted to text.',
                    {"column_name": column},
                ),
                "quickFixes": [
                    {
                        "text": i18n.trans(
                            "transpose.warnings.headersConvertedToText.quickFix.text",
                            "Convert {column_name} to text",
                            {"column_name": '"%s"' % column},
                        ),
                        "action": "prependModule",
                        "args": ["converttotext", {"colnames": [column]},],
                    }
                ],
            }
        )

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
        if to_convert:
            warnings.append(
                {
                    "message": i18n.trans(
                        "transpose.warnings.differentColumnTypes.message",
                        '{n_columns, plural, other {# columns (see "{first_colname}") were} one {Column "{first_colname}" was}} '
                        "auto-converted to Text because all columns must have the same type.",
                        {"n_columns": len(to_convert), "first_colname": to_convert[0]},
                    ),
                    "quickFixes": [
                        {
                            "text": i18n.trans(
                                "transpose.warnings.differentColumnTypes.quickFix.text",
                                "Convert {n_columns, plural, other {# columns} one {# column}} to text",
                                {"n_columns": len(to_convert)},
                            ),
                            "action": "prependModule",
                            "args": ["converttotext", {"colnames": to_convert},],
                        }
                    ],
                }
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

    if warnings:
        return (ret, warnings)
    else:
        return ret
# END copy/paste
