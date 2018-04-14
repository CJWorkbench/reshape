def render(table, params):
    import pandas as pd

    diridx = params['direction']
    cols = params.get('colnames', '')
    varcol = params.get('varcol', '')

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

        table = table.set_index(cols + [varcol]).unstack()

        cols = [col[-1] for col in table.columns.values]
        table.columns = cols        # get rid of multi-index hierarchy
        table = table.reset_index() # turn index cols into regular cols

    elif dir == 'transpose':
        # We assume that the first column is going to be the new header rowi
        new_columns = table[table.columns[0]].tolist()
        index_col = table.columns[0]
        table = table.set_index(index_col).transpose().reset_index()
        table.columns = new_columns

    return table

