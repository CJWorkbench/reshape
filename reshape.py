def render(table, params):
    import pandas as pd

    diridx = params['direction']
    cols = params.get('colnames', '')
    varcol = params.get('varcol', '')

    transpose_new_col_header = 'New Column'

    # no columns selected and not transpose, NOP
    if cols=='' and diridx != 2:
        return table
    cols = cols.split(',')

    dirmap = ['widetolong', 'longtowide', 'transpose']  # must match reshape.json
    dir = dirmap[diridx]

    if dir == 'widetolong':
        table = pd.melt(table, id_vars=cols)
        table.sort_values(cols, inplace=True)
        table = table.reset_index(drop=True)  # renumber after sort, don't add extra index col

    elif dir == 'longtowide':
        if varcol == '':            # gotta have this parameter
            return table

        keys = cols

        has_second_key = params.get('has_second_key', False)
        # If second key is used and present, append it to the list of columns
        if has_second_key:
            second_key = params.get('second_key', '')
            if second_key in table.columns:
                keys.append(second_key)

        table = table.set_index(keys + [varcol]).unstack()
        table.columns = [col[-1] for col in table.columns.values]
        table = table.reset_index()

    elif dir == 'transpose':
        # We assume that the first column is going to be the new header row
        # Use the content of the first column as the new headers
        # We set the first column header to 'New Column'. Using the old header is confusing.

        # Check if Column Header Exists in Column
        new_columns = table[table.columns[0]].tolist()
        suffix = 1
        while transpose_new_col_header in new_columns:
            if f'{transpose_new_col_header}_{str(suffix)}' not in new_columns:
                transpose_new_col_header = f'{transpose_new_col_header}_{str(suffix)}'
                break
            suffix += 1
        new_columns = [transpose_new_col_header] + new_columns
        index_col = table.columns[0]
        # Transpose table, reset index and correct column names
        table = table.set_index(index_col).transpose()
        # Clear columns in case CategoricalIndex dtype
        table.columns = ['']*len(table.columns)
        table = table.reset_index()
        table.columns = new_columns
        # Infer data type of each column (numeric or string)
        for col in table.columns:
            try:
                table[col] = pd.to_numeric(table[col], errors='raise')
            except:
                table[col] = table[col].astype(str)

    return table
