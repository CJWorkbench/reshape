import unittest
import pandas as pd
from reshape import render


class TestReshape(unittest.TestCase):
 
	def setUp(self):
		# this data is designed to sorted in the way that our wide to long operation would sort
		self.long1 = pd.DataFrame(
			{'date': ['2000-01-03', '2000-01-03', '2000-01-03', '2000-01-04', '2000-01-04', '2000-01-04', '2000-01-05', '2000-01-05', '2000-01-05', '2000-01-06', '2000-01-06', '2000-01-06'],
			 'variable':['George', 'Lisa', 'Michael', 'George', 'Lisa', 'Michael', 'George', 'Lisa', 'Michael', 'George', 'Lisa', 'Michael'],
			 'value':[200, 500, 450, 180.5, 450, 448, 177, 420, 447, 150, 300, 344.6]},
			 columns = ['date','variable','value'])

		self.wide1 = self.long1.set_index(['date','variable']).unstack()
		cols = [col[-1] for col in self.wide1.columns.values]
		self.wide1.columns = cols        # get rid of multi-index hierarchy
		self.wide1 = self.wide1.reset_index() # turn index cols into regular cols

		# Tables with more than one id column
		idcol = pd.Series(['a','b','c','d'])
		idcol.name = 'idcol'
		self.wide2 = pd.concat([idcol, self.wide1], axis=1)

		self.long2 = pd.melt(self.wide2, id_vars=['idcol','date']) 
		self.long2.sort_values(['idcol','date'], inplace=True)
		self.long2 = self.long2.reset_index(drop=True)  # renumber after sort, don't add extra index col


	def test_defaults(self):
		params = { 'direction': 0, 'colnames': '', 'varcol':''}
		out = render(self.wide1, params)
		self.assertTrue(out.equals(self.wide1)) # should NOP when first applied

	def test_wide_to_long(self):
		params = { 'direction': 0, 'colnames': 'date', 'varcol':''}
		out = render(self.wide1, params)
		self.assertTrue(out.equals(self.long1))

	def test_wide_to_long_mulicolumn(self):
		# two ID columns
		params = { 'direction': 0, 'colnames': 'idcol,date', 'varcol':''}
		out = render(self.wide2, params)
		self.assertTrue(out.equals(self.long2))

	def test_long_to_wide(self):
		params = { 'direction': 1, 'colnames': 'date', 'varcol':'variable'}
		out = render(self.long1, params)
		self.assertTrue(out.equals(self.wide1))

	def test_long_to_wide_missing_varcol(self):
		params = { 'direction': 1, 'colnames': 'date', 'varcol':''}
		out = render(self.long1, params)
		self.assertTrue(out.equals(self.long1)) # nop if no column selected

	def test_long_to_wide_mulicolumn(self):
		# two ID columns
		params = { 'direction': 1, 'colnames': 'idcol,date', 'varcol':'variable'}
		out = render(self.long2, params)
		self.assertTrue(out.equals(self.wide2))


if __name__ == '__main__':
		unittest.main()
