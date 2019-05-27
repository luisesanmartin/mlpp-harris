import sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier


def boosting(features, label, n=10):
	'''
    Returns an Ada Boosting classifier object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - n: the number of iterations for the classifier (default is 100)
	'''

	bc = AdaBoostClassifier(LogisticRegression(random_state=0, \
		solver='liblinear'), n_estimators=n)
	bc.fit(features, label)

	return bc


def bagging(features, label, n=1000, samples=0.2, features_size=1/3):
	'''
    Returns a bagging classifier object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - n: the number of classifiers in the model (default is 1000)
        - samples: the fraction representing the size of the samples
                   used for each model
        - features_size: the fraction represening the number of the features
                         used for each model, out of the total number of
                         features (default is 1/3)
	'''

	bagging = BaggingClassifier(KNeighborsClassifier(), \
		      max_samples=samples, max_features=features_size, n_estimators=n)
	bagging.fit(features, label)

	return bagging


def random_forest(features, label, n=1000, \
	features_size='auto', criteria='gini'):
	'''
    Returns a random forest model object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - n: the number of decision trees in the model (default is 1000)
        - features_size: the fraction represening the number of the features
                         used for each split, out of the total number of
                         features (default is sqrt(n_features))
        - criteria: node split criteria (default is gini)
	'''

	rf = RandomForestClassifier(random_state=0, n_estimators=n, \
		criterion=criteria, max_features=features_size)
	rf.fit(features, label)

	return rf


def svm(features, label, c_value=1.0):
	'''
    Returns a support vector machine classifier object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - c_value: penalty parameter of the error term (default is 1.0)
	'''

	svm = LinearSVC(random_state= 0, C=c_value)
	svm.fit(features, label)

	return svm


def logistic_regression(features, label, norm='l1', c_value=1.0):
	'''
    Returns a logistic regression object from sklearn

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - norm: norm used for penalization of overfitting (default is 'l1')
        - c_value: inverse of regularization stregnth (default is 1.0)
	'''

	lr = LogisticRegression(random_state=0, solver='liblinear', \
		penalty=norm, C=c_value)
	lr.fit(features, label)

	return lr


def decision_tree(features, label, depth=5, criteria='gini'):
	'''
    Returns a decision tree classifier object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - depth: max depth of the tree (default is 5)
        - criteria: split decision criteria (default is gini)
	'''

	dec_tree = DecisionTreeClassifier(max_depth=depth, criterion=criteria)
	dec_tree.fit(features, label)

	return dec_tree

def nearest_neighbors(features, label, k=3, distance='minkowski'):
	'''
    Returns a nearest neighbor classifier object from sklearn.

    Inputs:
        - features: a Pandas dataframe with the features
        - label: a Pandas series with the label variable
        - k: number of neighbors taken for classification (default is 3)
        - distance: distance measurement (default is minkowski)
	'''

	nn = KNeighborsClassifier(n_neighbors=k, metric=distance)
	nn.fit(features, label)

	return nn