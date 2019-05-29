import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import ParameterGrid


pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=Warning)

CLASSIFIERS = {'Ada boosting': AdaBoostClassifier,
               'Bagging': BaggingClassifier,
               'Random forest': RandomForestClassifier,
               'Support vector machine': LinearSVC,
               'Logistic regression': LogisticRegression,
               'Decision tree': DecisionTreeClassifier,
               'Nearest neighbors': KNeighborsClassifier}

PARAMETERS = \
{'Ada boosting': {'base_estimator': [LogisticRegression(C=10.0, penalty='l1'),
                                  LogisticRegression(C=1.0, penalty='l1'),
                                  LogisticRegression(C=0.1, penalty='l1'),
                                  LogisticRegression(C=0.01, penalty='l1'),
                                  LogisticRegression(C=10.0, penalty='l2'),
                                  LogisticRegression(C=1.0, penalty='l2'),
                                  LogisticRegression(C=0.1, penalty='l2'),
                                  LogisticRegression(C=0.01, penalty='l2'),
                                  LinearSVC(C=10.0, penalty='l1'),
                                  LinearSVC(C=1.0, penalty='l1'),
                                  LinearSVC(C=0.1, penalty='l1'),
                                  LinearSVC(C=0.01, penalty='l1'),
                                  LinearSVC(C=10.0, penalty='l2'),
                                  LinearSVC(C=1.0, penalty='l2'),
                                  LinearSVC(C=0.1, penalty='l2'),
                                  LinearSVC(C=0.01, penalty='l2')],
                  'n_estimators': [30, 50, 100]},
 'Bagging': {'base_estimator': [DecisionTreeClassifier(max_depth=1, criterion='gini'),
                             DecisionTreeClassifier(max_depth=5, criterion='gini'),
                             DecisionTreeClassifier(max_depth=10, criterion='gini'),
                             DecisionTreeClassifier(max_depth=1, criterion='entropy'),
                             DecisionTreeClassifier(max_depth=15, criterion='entropy'),
                             DecisionTreeClassifier(max_depth=10, criterion='entropy'),
                             KNeighborsClassifier(n_neighbors=3),
                             KNeighborsClassifier(n_neighbors=10),
                             KNeighborsClassifier(n_neighbors=30)],
             'n_estimators': [10, 100, 500],
             'max_samples': [1/5, 1/3, 1/2],
             'max_features': [1/5, 1/3, 1/2]},
 'Random forest': {'n_estimators': [100, 1000, 5000],
                   'criterion': ['gini', 'entropy'],
                   'max_features': [0.1, 0.2, 1/3]},
 'Support vector machine': {'C': [0.001, 0.01, 0.1, 1, 10],
                            'penalty': ['l1', 'l2']},
 'Logistic regression': {'C': [0.001, 0.01, 0.1, 1, 10],
                         'penalty': ['l1', 'l2']},
 'Decision tree': {'max_depth': [1, 5, 10],
                   'criterion': ['gini', 'entropy'],
                   'min_samples_split': [10, 100, 1000]},
 'Nearest neighbors': {'n_neighbors': [3, 10, 30]}}


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

    df['label'] = np.where(df[date2_col] > df[date1_col] + days, 1, 0)

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

def get_predictions(classifier, X_test):
    '''
    Returns a Pandas Series with the prediction scores.

    Inputs:
        - classifier object
        - X_test: test dataset (Pandas)
    Output: Pandas series with the prediction scores
    '''

    if hasattr(classifier, 'predict_proba'):
        pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    else:
        pred_scores = pd.Series(classifier.decision_function(X_test))

    return pred_scores

def simple_classifier(y_test):
    '''
    Returns a float number with the accuracy if we just predicted every
    value in the test set to be 1/0, whatever fraction is higher in y_test.

    Inputs:
        y_test: Pandas series with the test label
    Output: accuracy of this simple classifier method
    '''

    mean = y_test.mean()

    if mean >= 0.5:
        print("Predicting every data point's value to be 1, " + \
              "the accuracy is", round(mean*100, 1), "%")

        return mean

    else:
        acc = 1 - mean
        print("Predicting every data point's value to be 0, " + \
              "the accuracy is", round(acc*100, 1), "%")

        return acc

def accuracy(classifier, threshold, X_test, y_test):
    '''
    Returns the accuracy (float) of a classifier given a certain threshold,
    and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: accuracy (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_scores.sort_values(ascending=False, inplace=True)
    pred_scores.reset_index(drop=True, inplace=True)
    pred_label = np.where(pred_scores.index + 1 <= \
                          threshold * len(y_test), 1, 0)
    acc = accuracy_score(pred_label, y_test)

    return acc

def precision(classifier, threshold, X_test, y_test):
    '''
    Returns the precision (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: precision (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_scores.sort_values(ascending=False, inplace=True)
    pred_scores.reset_index(drop=True, inplace=True)
    pred_label = np.where(pred_scores.index + 1 <= \
                          threshold * len(y_test), 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    prec = true_positives / (false_positive + true_positives)

    return prec

def recall(classifier, threshold, X_test, y_test):
    '''
    Returns the recall (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: recall (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_scores.sort_values(ascending=False, inplace=True)
    pred_scores.reset_index(drop=True, inplace=True)
    pred_label = np.where(pred_scores.index + 1 <= \
                          threshold * len(y_test), 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    rec = true_positives / (false_negatives + true_positives)

    return rec

def f1(classifier, threshold, X_test, y_test):
    '''
    Returns the f1 score (float) of a classifier given a certain
    threshold, and a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - threshold: a fraction that denotes the upper percent of the
                     population that will have positively predicted labels        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: f1 score (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    pred_scores.sort_values(ascending=False, inplace=True)
    pred_scores.reset_index(drop=True, inplace=True)
    pred_label = np.where(pred_scores.index + 1 <= \
                          threshold * len(y_test), 1, 0)
    score = f1_score(y_test, pred_label)

    return score

def area_under_curve(classifier, X_test, y_test):
    '''
    Returns the area under the curve (float) of a classifier
    given a certain test set (X_test and y_test).

    Inputs:
        - classifier: the model we are using
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: area under the curve (float)
    '''

    pred_scores = get_predictions(classifier, X_test)
    fpr, tpr, _ = roc_curve(y_test, pred_scores, pos_label=1)
    area = auc(fpr, tpr)

    return area

def precision_recall_curves(classifier, X_test, y_test):
    '''
    (This function uses code borrowed from the lab 4)

    Plots the precision and recall curves of a classifier, given a
    certain test set.

    Inputs:
        - classifier: the model we are using
        - X_test: a Pandas dataframe with the features of the test set
        - y_test: a Pandas series with the label of the test set
    Output: plot object
    '''

    pred_scores = get_predictions(classifier, X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, \
                                    pred_scores,pos_label=1)
    population = [1.*sum(pred_scores>=threshold)/len(pred_scores) \
                 for threshold in thresholds]
    p, = plt.plot(population, precision[:-1], color='b')
    r, = plt.plot(population, recall[:-1], color='r')
    plt.legend([p, r], ['precision', 'recall'])
    
    return plt

def evaluation_table(classifiers, parameters, datasets, fractions, \
                     features, label):
    '''
    (Please notice that this function might take a while to run)

    Returns a dataframe where each row is a classifier from classifiers
    and each column is a model performance indicator. Each classifier
    is evaluated on the same features and the same label.

    Inputs:
        - classifiers: a dictionary with the (untrained) classifiers 
                       we want to use
        - parameters: a dictionary with the parameters we want to try out.
                      Each key must be associated with a key from the
                      classifiers dictionary
        - datasets: a dictionary of pairs of datasets - training and
                    testing sets
        - fractions: a list of floats where each number denotes the upper
                     percent of the population for which the precision and
                     recall will be evaluated
        - features: the list of features we want to use for all models
        - label: the label we want to use for all models

    Output: a Pandas dataframe - the evaluation table
    '''

    # Generating the df
    precision_cols = ['precision_at_' + str(i) for i in fractions]
    recall_cols = ['recall_at_' + str(i) for i in fractions]
    df = pd.DataFrame(columns=['Exact classifier', 'classifier', \
                               'parameters', 'dataset','baseline'] \
                               + precision_cols + recall_cols + ['AUC ROC'])
    
    # Counting the total number of models -- for on-the-run progress reporting
    total = 0
    for clf in classifiers:
        clf_n = 1
        for parameter in parameters[clf]:
            clf_n = clf_n * len(parameters[clf][parameter])
        total += clf_n
    
    # Starting with the loop
    i = 1
    for dataset in datasets:

        # Generating datasets
        train_set, test_set = datasets[dataset]
        train_X = train_set[features]
        train_y = train_set[label]
        test_X = test_set[features]
        test_y = test_set[label]

        baseline = simple_classifier(test_y)

        for classifier in classifiers:

            parameters_list = list(ParameterGrid(parameters[classifier]))

            for parameter in parameters_list:

                # Progress reporting
                print('\nRunning model', i, 'out of', total)
                print('Progress:', round(i/total*100, 1), '%')
                i += 1

                # Estimating models and metrics
                clf = classifiers[classifier](**parameter)
                model = clf.fit(train_X, train_y)
                precision_metrics = [precision(model, fraction, 
                                     test_X, test_y) for fraction in fractions]
                recall_metrics = [recall(model, fraction, test_X, test_y) \
                                  for fraction in fractions]

                # Appending results
                df.loc[len(df)] = [str(clf), classifier, parameter, dataset,
                                   baseline] + precision_metrics + \
                                   recall_metrics + \
                                   [area_under_curve(model, test_X, test_y)]
    
    return df