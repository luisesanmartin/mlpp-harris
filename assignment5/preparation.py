import pandas as pd
import numpy as np


def create_time_label(df, date_posted, date_funded):
    '''
    '''

    days60 = pd.DateOffset(days=60)

    df['funded'] = np.where(df[date_funded] <= df[date_posted] + days60, 1, 0)


def time_based_split(df, time_col, date_threshold, months_range):
    '''
    '''

    date_lower_threshold = pd.to_datetime(date_threshold)
    date_upper_threshold = date_lower_threshold + \
                           pd.DateOffset(months=months_range)
    df_train = df[df[time_col]<=date_lower_threshold]
    df_test = df[(df[time_col]>date_lower_threshold) \
              & (df[time_col]<=date_upper_threshold)]

    print('train/test threshold:', date_lower_threshold)
    print('test upper threshold:', date_upper_threshold)

    return df_train, df_test


def to_date(df, column):
    '''
    '''

    df[column] = pd.to_datetime(df[column], infer_datetime_format=True)


def discrete_0_1(df, column, value0, value1):
    '''
    '''

    df[column] = df[column].replace(value0, 0)
    df[column] = df[column].replace(value1, 1)
    df[column] = pd.to_numeric(df[column])


def fill_nas_other(df, column, label):
    '''
    '''

    df[column] = df[column].fillna(value=label)


def fill_nas_mode(df, column):
    '''
    '''

    mode = df[column].mode().iloc[0]
    df[column] = df[column].fillna(value=mode)


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


def discretize(df, column):
    '''
    Creates in the dataframe provided dummy variables indicating that an
    observation belongs to a certain quartile of the column provided.
    Each dummy has the name of the column + a number indicating the quartile.

    Inputs:
        - column (column of a pandas dataframe): the column we want to
        discretize. It should be a continuous variable included in df.
        - df: the pandas dataframe where column is and where we'll add
        the new dummy variables.

    Output: nothing. Modifies the df directly.
    '''
    N_SUBSETS = 4
    WIDE = 1 / N_SUBSETS
    
    xtile = 0
    col = df[column]

    for i in range(1, N_SUBSETS + 1):

        mini = col.quantile(xtile)
        maxi = col.quantile(xtile + WIDE)
        df.loc[(df[column] >= mini) & (df[column] <= maxi), \
               column + '_quartile'] = i
        xtile += WIDE


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


def replace_over_one(df, column):
    '''
    Takes a dataframe (df) and a variable in it (column) and replaces
    the values over one with ones.

    Inputs:
        - column (column of a pandas dataframe): the column whose values
        over one we will replace with ones.
        - df: the pandas dataframe where column is.
    Output: nothing. Modifies the df directly.
    '''

    df.loc[df[column] > 1, column] = 1


def discretize_over_zero(df, column):
    '''
    Takes a dataframe (df) and a variable in it (column) and creates a
    dummy indicating the observations that have a value higher than
    zero.

    Inputs:
        - column (column of a pandas dataframe): the column whose values
        we'll take to create the dummy.
        - df: the pandas dataframe where column is.
    Output: nothing. Modifies the df directly.
    '''

    df.loc[df[column] == 0, column + '_over_zero'] = 0
    df.loc[df[column] > 0, column + '_over_zero'] = 1
