import pandas as pd


def read(csv_file):
    '''
    Reads the data of a csv file and returns a Pandas dataframe of it.

    Inputs:
        - csv_file: a csv file containing the dataframe
    '''

    return pd.read_csv(csv_file)


def columns_list(df):
    '''
    Prints a list of the columns of a dataframe. No output is returned.

    Inputs:
        - df: Pandas dataframe
    '''

    print(df.columns)


def count_obs(df):
    '''
    Prints the number of observations (rows) in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    print(len(df))


def columns_types(df):
    '''
    Prints the type of all columns in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    print(df.dtypes)


def count_missings(df):
    '''
    Prints the percentage of missing data points in each column of a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    total = len(df)
    for col in df.columns:
        print(col, 'has', df[col].isna().sum() / total * 100, \
            '% of missing data points')


def tabulate(df, col):
    '''
    Tabulates the column (col) of a dataframe (df), and prints the result.

    Inputs:
        - df: Pandas dataframe
        - col: The column we want to tabulate
    '''

    print(df.groupby(col).size())


def correlations(df):
    '''
    Prints the correlation table of all the columns in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    df.corr().style.background_gradient(cmap='coolwarm')


def histograms(df):
    '''
    Prints a histogram of every column in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    df.hist(figsize=(20, 20))


def describe(df):
    '''
    Prints some descriptive statistics of all the columns of a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    for col in df.columns:
        print(df[col].describe())


def duplicates(df):
    '''
    Prints the number of duplicated observations (all-columns-duplicates)
    in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    dups = df[df.duplicated(keep=False)]
    print(len(dups))


def duplicates_in_columns(df, columns):
    '''
    Prints the number of observations with the same values for a certain
    group of columns in a dataframe.

    Inputs:
        - df: Pandas dataframe
        - columns: a group of columns in the dataframe.
    '''

    dups = df[df.duplicated(columns, keep=False)]
    print(len(dups))