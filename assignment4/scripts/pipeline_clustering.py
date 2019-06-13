import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def read(csv_file):
    '''
    Reads the data of a csv file and returns a Pandas dataframe of it.

    Inputs:
        - csv_file: a csv file containing the dataframe
    '''

    return pd.read_csv(csv_file)

def fill_nas_median(df, column):
    '''
    Replaces the NaN values of a column (column) in a dataframe (df) with
    the value of the column median

    Inputs:
        - column (column of a pandas dataframe): the column whose NaN
        values we want to fill in with the median. It should be a variable
        included in df.
        - df: the pandas dataframe where column is and where we'll replace
        the NaN values.

    Output: nothing. Modifies the df directly.
    '''

    median = df[column].quantile()
    df[column] = df[column].fillna(value=median)

def fill_nas_other(df, column, label):
    '''
    Fills the NaN values of a column (column) in a dataframe (df) with
    the value provided (label).

    Inputs:
        - column (column of a pandas dataframe): the column whose NaN
        values we want to fill in. It should be a variable
        included in df
        - df: the pandas dataframe where column is and where we'll replace
        the NaN values

    Output: nothing. Modifies the df directly.
    '''

    df[column] = df[column].fillna(value=label)

def columns_list(df):
    '''
    Prints a list of the columns of a dataframe. No output is returned.

    Inputs:
        - df: Pandas dataframe
    '''

    for col in df.columns:
        print(col)

def columns_types(df):
    '''
    Prints the type of all columns in a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    print(df.dtypes)

def create_dummies(df, column):
    '''
    Takes a dataframe (df) and a categorical variable in it (column) and
    creates a dummy for each distinct value of the input categorical
    variable.

    Inputs:
        - column (column of a pandas dataframe): the column we want to
        discretize. It should be a categorical variable included in df.
        - df: the pandas dataframe where column is and where we'll add
        the new dummy variables
    Output: nothing. Modifies the df directly.       
    '''

    for value in df[column].unique():

        df.loc[df[column] == value, column + '_' + str(value)] = 1
        df.loc[df[column] != value, column + '_' + str(value)] = 0

def add_cluster_to_df(n_clusters, df, X, new_col_name):
    '''
    '''

    kmeans = KMeans(n_clusters, random_state=0).fit(df[X])
    df[new_col_name] = pd.Series(kmeans.labels_)

def tabulate(df, col):
    '''
    Tabulates the column (col) of a dataframe (df), and prints the result.

    Inputs:
        - df: Pandas dataframe
        - col: The column we want to tabulate
    '''

    print(df.groupby(col).size())

def plot_cluster_and_two_variables(df, cluster_col, var1, var2):
    '''
    '''

    groups = df.groupby(cluster_col)
    fig, ax = plt.subplots()
    for pred_class, group in groups:
        ax.scatter(group[var1], group[var2], label=pred_class)
    ax.legend()
    return plt

def describe(df):
    '''
    Prints some descriptive statistics of all the columns of a dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    for col in df.columns:
        print(df[col].describe())

def histogram(df, col):
    '''
    Prints a histogram of the column 'col' in the dataframe.

    Inputs:
        - df: Pandas dataframe
    '''

    df.hist(column=col)