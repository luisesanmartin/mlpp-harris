import pandas as pd


def read(csv_file):
    '''
    '''

    return pd.read_csv(csv_file)


def columns_list(df):
	'''
	'''

	print(df.columns)


def count_obs(df):
    '''
    '''

    print(len(df))


def columns_types(df):
	'''
	'''

	print(df.dtypes)


def count_missings(df):
	'''
	'''

	total = len(df)
	for col in df.columns:
		print(col, 'has', df[col].isna().sum() / total * 100, \
			'% of missing data points')


def tabulate(df, col):
	'''
	'''

	print(df.groupby(col).size())


def correlations(df):
	'''
	'''

	df.corr().style.background_gradient(cmap='coolwarm')


def histograms(df):
	'''
	'''

	df.hist(figsize=(20, 20))


def describe(df):
	'''
	'''

	for col in df.columns:
		print(df[col].describe())


def duplicates(df):
	'''
	'''

	dups = df[df.duplicated(keep=False)]
	print(len(dups))


def duplicates_in_columns(df, columns):
	'''
	'''

	dups = df[df.duplicated(columns, keep=False)]
	print(len(dups))
