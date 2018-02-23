def render(table, params):
    import pandas as pd

    diridx = params['direction']
    cols = params['colnames']
    varcol = params['varcol']

    # no columns selected, NOP
    if cols=='':
        return table
    cols = cols.split(',')

    dirmap = ['widetolong', 'longtowide']  # must match reshape.json
    dir = dirmap[diridx]

    if dir == 'widetolong':
        table = pd.melt(table, id_vars=cols)
        table.sort_values(cols, inplace=True)
        table = table.reset_index(drop=True)  # renumber after sort, don't add extra index col

    else: # longtowide
        if varcol == '':            # gotta have this parameter
            return table       

        table = table.set_index(cols + [varcol]).unstack()

        cols = [col[-1] for col in table.columns.values]
        table.columns = cols        # get rid of multi-index hierarchy
        table = table.reset_index() # turn index cols into regular cols

    return table

