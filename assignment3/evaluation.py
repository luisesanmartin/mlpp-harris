import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def simple_classifier(y_test):
    '''
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
    '''

    pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    pred_label = np.where(pred_scores > threshold, 1, 0)
    acc = accuracy_score(pred_label, y_test)

    return acc


def precision(classifier, threshold, X_test, y_test):
    '''
    '''

    pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    pred_label = np.where(pred_scores > threshold, 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    prec = 1.0 * true_positives / (false_positive + true_positives)

    return prec


def recall(classifier, threshold, X_test, y_test):
    '''
    '''

    pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    pred_label = np.where(pred_scores > threshold, 1, 0)
    c = confusion_matrix(y_test, pred_label)
    true_negatives, false_positive, false_negatives, true_positives = c.ravel()
    rec = 1.0 * true_positives / (false_negatives + true_positives)

    return rec


def f1(classifier, threshold, X_test, y_test):
    '''
    '''

    pred_scores = pd.Series(classifier.predict_proba(X_test)[:,1])
    pred_label = np.where(pred_scores > threshold, 1, 0)
    score = f1_score(y_test, pred_label)

    return score


def area_under_curve(classifier, X_test, y_test):
    '''
    '''

    pred_scores = classifier.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, pred_scores, pos_label=1)
    area = auc(fpr, tpr)

    return area


def precision_recall_curves(classifier, X_test, y_test):
    '''
    Important: This function uses code borrowed from the lab 4
    '''

    pred_scores = classifier.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, \
                                    pred_scores,pos_label=1)
    population = [1.*sum(pred_scores>threshold)/len(pred_scores) \
                 for threshold in thresholds]
    p, = plt.plot(population, precision[:-1], color='b')
    r, = plt.plot(population, recall[:-1], color='r')
    plt.legend([p, r], ['precision', 'recall'])
    
    return plt

def evaluation_table(classifiers, X_test, y_test):
    '''
    '''

    df = pd.DataFrame()
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, \
                  0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    df['classifier'] = classifiers
    df['baseline'] = [simple_classifier(y_test)]*len(df)

    for metric in ['precision', 'recall']:
        
        for threshold in thresholds:

            l = []
        
            for classifier in classifiers:

                if metric == 'precision':

                    l.append(precision(classifier, threshold, X_test, y_test))

                else:

                    l.append(recall(classifier, threshold, X_test, y_test))

            df[metric + '_' + str(threshold)] = l

    auc_values = []
    for classifier in classifiers:

        auc_values.append(area_under_curve(classifier, X_test, y_test))
    df['auc_roc'] = auc_values

    return df

                