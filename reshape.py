from typing import Iterator, List, Set
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
from cjwmodule import i18n


MAX_N_COLUMNS = 100


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
def transpose(table, params, *, input_columns):
    warnings = []
    colnames_auto_converted_to_text = []

    if len(table) > MAX_N_COLUMNS:
        table = table.truncate(after=MAX_N_COLUMNS - 1)
        warnings.append(i18n.trans(
            "transpose.warnings.tooManyRows",
            "We truncated the input to {max_columns} rows so the "
            "transposed table would have a reasonable number of columns.",
            {"max_columns": MAX_N_COLUMNS}
        ))

    if not len(table.columns):
        # happens if we're the first module in the module stack
        return pd.DataFrame()

    # If user does not supply a name (default), use the input table's first
    # column name as the output table's first column name.
    first_colname = params["firstcolname"].strip() or table.columns[0]

    column = table.columns[0]
    headers_series = table[column]
    table.drop(column, axis=1, inplace=True)

    # Ensure headers are string. (They will become column names.)
    if input_columns[column].type != "text":
        warnings.append({
            "message": i18n.trans(
                "transpose.headersConvertedToText.error",
                'Headers in column "{column_name}" were auto-converted to text.',
                {"column_name": column}
            ),
            "quickFixes": [
                {
                    "text": i18n.trans(
                        "transpose.headersConvertedToText.quick_fix.text",
                        "Convert {column_name} to text",
                        {"column_name": '"column"'}
                    ),
                    "action": "prependModule",
                    "args": [
                        "converttotext",
                        {"colnames": column},
                    ],
                }
            ]
        })

    # Regardless of column type, we want to convert to str. This catches lots
    # of issues:
    #
    # * Column names shouldn't be a CategoricalIndex; that would break other
    #   Pandas functions. See https://github.com/pandas-dev/pandas/issues/19136
    # * nulls should be converted to '' instead of 'nan'
    # * Non-str should be converted to str
    # * `first_colname` will be the first element (so we can enforce its
    #   uniqueness).
    #
    # After this step, `headers` will be a List[str]. "" is okay for now: we'll
    # catch that later.
    na = headers_series.isna()
    headers_series = headers_series.astype(str)
    headers_series[na] = ""  # Empty values are all equivalent
    headers_series[headers_series.isna()] = ""
    headers = headers_series.tolist()
    headers.insert(0, first_colname)
    non_empty_headers = [h for h in headers if h]

    # unique_headers: all the "valuable" header names -- the ones we won't
    # rename any duplicate/empty headers to.
    unique_headers = set(headers)

    if "" in unique_headers:
        warnings.append(i18n.trans(
            "transpose.warnings.renamedColumnsDueToEmpty",
            'We renamed some columns because the input column "{column}" had '
            "empty values.",
            {"column": column}
        ))
    if len(non_empty_headers) != len(unique_headers - set([""])):
        warnings.append(i18n.trans(
            "transpose.warnings.renamedColumnsDueToDuplicate",
            'We renamed some columns because the input column "{column}" had '
            "duplicate values.",
            {"column": column}
        ))

    headers = list(_uniquize_colnames(headers, unique_headers))

    table.index = headers[1:]

    input_types = set(c.type for c in input_columns.values() if c.name != column)
    if len(input_types) > 1:
        # Convert everything to text before converting. (All values must have
        # the same type.)
        to_convert = [c for c in table.columns if input_columns[c].type != "text"]
        cols_str = ", ".join(f'"{c}"' for c in to_convert)
        warnings.append({
            "message": i18n.trans(
                "transpose.differentColumnTypes.error",
                "{n_columns, plural, other {Columns {column_names} were} one {Column {column_names} was}} "
                "auto-converted to Text because all columns must have the same type.",
                {
                    "n_columns": len(to_convert),
                    "column_names": cols_str
                }
            ),
            "quickFixes":[
                {
                    "text": i18n.trans(
                        "transpose.warnings.differentColumnTypes.quick_fix.text",
                        "Convert {column_names} to text",
                        {"column_names": cols_str}
                    ),
                    "action": "prependModule",
                    "args": [
                        "converttotext",
                        {"colnames": ",".join(to_convert)},
                    ],
                }
            ]
        })

        for colname in to_convert:
            # TODO respect column formats ... and nix the quick-fix?
            na = table[colname].isnull()
            table[colname] = table[colname].astype(str)
            table[colname][na] = np.nan

    # The actual transpose
    ret = table.T
    # Set the name of the index: it will become the name of the first column.
    ret.index.name = first_colname
    # Make the index (former colnames) a column
    ret.reset_index(inplace=True)

    if warnings:
        return (ret, warnings)
    else:
        return ret


def _uniquize_colnames(
    colnames: Iterator[str], never_rename_to: Set[str]
) -> Iterator[str]:
    """
    Rename columns to prevent duplicates or empty column names.

    The algorithm: iterate over each `colname` and add to an internal "seen".
    When we encounter a colname we've seen, append " 1", " 2", " 3", etc. to it
    until we encounter a colname we've never seen that is not in
    `never_rename_to`.
    """
    seen = set()
    for colname in colnames:
        force_add_number = False
        if not colname:
            colname = "unnamed"
            force_add_number = "unnamed" in never_rename_to
        if colname in seen or force_add_number:
            for i in range(1, 999999):
                try_colname = f"{colname} {i}"
                if try_colname not in seen and try_colname not in never_rename_to:
                    colname = try_colname
                    break

        seen.add(colname)
        yield colname


# END copy/paste
