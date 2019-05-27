import pandas as pd
import numpy as np


def create_time_label(df, date1_col, date2_col, n_days):
    '''
    Creates a dummy column named 'label' indicating if the value of
    date2_col is higher than date1_col + n_days.

    Inputs:
        - df: the Pandas dataframe we are using
        - date1_col: a date type column
        - date2_col: another date type column
        - n_days: the number of days we use to create the label

    Output: nothing, just modifies the df directly.
    '''

    days = pd.DateOffset(days=n_days)

    df['label'] = np.where(df[date_funded] > df[date_posted] + days, 1, 0)


def time_based_split(df, time_col, date_threshold, gap_days, months_range):
    '''
    Generates and returns a train and a test dataframe based on a time split
    and a gap period.

    Inputs:
        - df: the Pandas dataframe we want to generate the new dataframes from
        - time_col: the time column we will use from df for the time split
        - date_threshold: the date which will divide the train and test
                          dataframes
        - gap_days: the number of days we use for a gap period before the
                    date threhold
        - months_range: the number of months we want our test dataset to span

    Outputs:
        - df_train: the training dataset, based on the split. It consists of
                    all data point where time_col is equal or lower than
                    date_threshold minus gap_days
        - df_test: the testing dataset. It has all observations where time_col
                   is higher than date_threshold and equal or lower than
                   date_threshold plus months_range
    '''

    gap = pd.DateOffset(days=gap_days)
    months = pd.DateOffset(months=months_range)
    date_split = pd.to_datetime(date_threshold)

    train_upper_threshold = date_split - gap
    test_upper_threshold = date_split + months

    df_train = df[df[time_col]<=train_upper_threshold]
    df_test = df[(df[time_col]>date_split) \
              & (df[time_col]<=test_upper_threshold)]

    print('train upper threshold:', train_upper_threshold)
    print('Notice that we leave a gap of', gap_days, 'days')
    print('test lower threshold:', date_split)
    print('test upper threshold:', test_upper_threshold)

    return df_train, df_test


def to_date(df, column):
    '''
    Transforms a column (column) of a dataframe (df) in date type.

    Inputs:
        - column (column of a pandas dataframe): the column whose type
        we want to replace
        - df: the pandas dataframe where column is

    Output: nothing. Modifies the df directly.
    '''

    df[column] = pd.to_datetime(df[column], infer_datetime_format=True)


def discrete_0_1(df, column, value0, value1):
    '''
    Replaces the value provided in 'value0' for 0 and the value of 'value1'
    for 1. Then transforms the column to a numeric type.

    Inputs:
        - df: the pandas dataframe we want to modify
        - column: the columns whose values we want to replace
        - value0: the value we should replace for zeros
        - value1: the value we should replace for ones

    Output: nothing. Modifies the df directly.
    '''

    df[column] = df[column].replace(value0, 0)
    df[column] = df[column].replace(value1, 1)
    df[column] = pd.to_numeric(df[column])


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


def fill_nas_mode(df, column):
    '''
    Fills the NaN values of a column (column) in a dataframe (df) with
    the value of the column mode.

    Inputs:
        - column (column of a pandas dataframe): the column whose NaN
        values we want to fill in with the mode. It should be a variable
        included in df
        - df: the pandas dataframe where column is and where we'll replace
        the NaN values

    Output: nothing. Modifies the df directly.
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
