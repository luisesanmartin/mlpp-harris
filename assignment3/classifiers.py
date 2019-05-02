import sklearn.tree as tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostRegressor


def boosting(features, label, n=10):
	'''
	'''

	br = AdaBoostRegressor(LogisticRegression(random_state=0, \
		solver='liblinear'), n_estimators=n)
	br.fit(features, label)

	return br


def bagging(features, label, n=10, samples=0.2, features_size=1/3):
	'''
	'''

	bagging = BaggingClassifier(KNeighborsClassifier(), \
		      max_samples=samples, max_features=features_size, n_estimators=n)
	bagging.fit(features, label)

	return bagging


def random_forest(features, label, n=10, \
	features_size='auto', criteria='gini'):
	'''
	'''

	rf = RandomForestClassifier(random_state=0, n_estimators=n, \
		criterion='criteria', max_features=features_size)
	rf.fit(features, label)

	return rf


def svm(features, label, c_value=1.0):
	'''
	'''

	svm = LinearSVC(random_state= 0, C=c_value)
	svm.fit(features, label)

	return svm


def logistic_regression(features, label, norm='l1', c_value=1.0):
	'''
	'''

	lr = LogisticRegression(random_state=0, solver='liblinear', \
		penalty=norm, C=c_value)
	lr.fit(features, label)

	return lr


def decision_tree(features, label, depth=5, criteria='gini'):
	'''
	'''

	dec_tree = DecisionTreeClassifier(max_depth=depth, criterion=criteria)
	dec_tree.fit(features, label)

	return dec_tree

def nearest_neighbors(features, label, k=3, distance='minkowski'):
	'''
	'''

	nn = KNeighborsClassifier(n_neighbors=k, metric=distance)
	nn.fit(features, label)

	return nn