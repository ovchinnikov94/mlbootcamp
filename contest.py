# -*- coding: utf-8 -*-
__author__ = 'Dmitriy Ovchinnikov'

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut
from sklearn.decomposition import PCA
from mlxtend.regressor import StackingRegressor

def prepare(x, y=None):
	data = x
	categorial = [c for c in data.columns if data[c].dtype.name == 'object']
	numerical = [c for c in data.columns if data[c].dtype.name != 'object']
	data_describe = data.describe(include=[object])
	data_categorial = pd.get_dummies(data[categorial])
	data_numerical = data[numerical]
	data_numerical = (data_numerical - data_numerical.mean()) / (data_numerical.max() - data_numerical.min())
	data_numerical.fillna(0, inplace=True)
	data = pd.concat((data_numerical, data_categorial), axis=1)
	data = pd.DataFrame(data, dtype = float)
	return data


if __name__ == '__main__':

	target = pd.read_csv('y_train.csv')
	data = prepare(pd.read_csv('x_train.csv'), target)
	#print(data.corr())

	test = prepare(pd.read_csv('x_test.csv'))

	#clf = linear_model.RidgeCV(alphas=[6.411, 6.411, 6.413], cv=20)
	#clf = linear_model.LassoCV(alphas=[0.819, 0.819, 0.821], max_iter=500, random_state=11, cv=20, n_jobs=-1, verbose=1)
	#clf = ensemble.RandomForestRegressor(n_estimators=300, max_features='auto', n_jobs=-1, verbose=1)
	#clf = neighbors.KNeighborsRegressor(n_jobs=-1, n_neighbors=2)

	#clf = ensemble.GradientBoostingRegressor(max_depth=5, n_estimators=200) #0.142
	#clf = ensemble.BaggingRegressor(
		#ensemble.RandomForestRegressor(n_estimators=300, max_features='auto', n_jobs=-1, verbose=1)) #0.384
	#clf = ensemble.AdaBoostRegressor(
		#base_estimator=ensemble.RandomForestRegressor(n_estimators=150, max_features='auto', n_jobs=-1),
		#n_estimators=100, loss='linear') #?
	#Adaboost + ExtraTrees
	#TruncatedSVD?
	'''adaboost = ensemble.AdaBoostRegressor(
		base_estimator=ensemble.ExtraTreesRegressor(n_estimators=100, n_jobs=-1, verbose=1),
		n_estimators=150, loss='linear')
	svd = TruncatedSVD(n_components=100, algorithm='arpack')

	clf = Pipeline([
		('svd', svd),
		('boost', adaboost)
		])
	'''
	#Gradient with defferent parametres
	#clf = ensemble.GradientBoostingRegressor(max_depth=20, n_estimators=300)

	#MLPRegressor - only 0.18 version of sklearn

	'''clf = ensemble.BaggingRegressor(
		base_estimator=ensemble.GradientBoostingRegressor(max_depth=5, n_estimators=100),
		n_jobs=-1
		)
	'''
	#clf = ensemble.AdaBoostRegressor(n_estimators=1000, loss='exponential')

	#clf = ensemble.RandomForestRegressor(n_estimators=1500, max_features='auto', n_jobs=-1, verbose=1) #0.143

	#clf = ensemble.ExtraTreesRegressor(n_jobs=-1, verbose=1, n_estimators=1500) #0.111

	#clf = SVR(kernel='rbf', C=1e3, gamma=0.5) #0.096
	#clf = ensemble.AdaBoostRegressor(base_estimator=SVR(kernel='rbf', C=1e3, gamma=1.0), n_estimators=500, loss='exponential')

	#clf = SVR(verbose=True, kernel='rbf', degree=2, gamma=0.8, C=1.0, epsilon=0.01)

	'''clf = ensemble.BaggingRegressor(
		base_estimator=ensemble.RandomForestRegressor(n_jobs=-1, verbose=1, n_estimators=3),
		n_estimators=800
	)'''

	'''model = Pipeline([
		('decomp', PCA()),
		('svr', SVR(kernel='rbf'))
	])

	parametres = {
		'decomp__n_components' : [2, 5, 10],
		'svr__C' : [0.5, 1.0, 5.0, 10.0, 15.0],
		'svr__gamma' : [0.1, 0.4, 0.8, 1.2],
		'svr__epsilon' : [0.5, 0.1, 0.01]
	}

	clf = GridSearchCV(estimator=model, param_grid=parametres, n_jobs=-1, cv = LeaveOneOut(6), verbose=1)
	'''
	'''extrees = ensemble.ExtraTreesRegressor(n_estimators=1000, n_jobs=-1, verbose=1)
	rforest = ensemble.RandomForestRegressor(n_estimators=1000, max_features='auto', n_jobs=-1, verbose=1)
	gradient = ensemble.GradientBoostingRegressor(n_estimators=1000, max_depth=10)
	svr1 = SVR(kernel='rbf', C=1e3, gamma=0.7)
	svr2 = SVR(kernel='rbf', C=1e3, gamma=0.5)
	svr3 = SVR(kernel='rbf', C=1e3, gamma=0.3)
	svr4 = SVR(kernel='rbf', C=1e3, gamma=0.1)

	clf = StackingRegressor(regressors=[svr1, svr2, svr3, svr4], meta_regressor=gradient, verbose=1)

	'''
	parametres = {
		'C' : np.arange(0.26, 0.32, 0.005)
	}

	#clf = SVR(kernel='rbf', C=1e3, gamma=0.28)
	clf = GridSearchCV(estimator=SVR(kernel='rbf', C=1e3), param_grid=parametres, n_jobs=-1, cv=20, verbose=1)

	clf.fit(data, target['time'])
	#print(clf.feature_importances_)
	result = clf.predict(test)
	best_parametres, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
	for param_name in sorted(parametres.keys()):
		print("%s: %r" % (param_name, best_parametres[param_name]))
	print("Score: %0.4f" % score)




	result = list(map(lambda x : 1.0 if x < 0.0 else x,result))

	f = open('results.txt', 'w+')
	for num in result:
		f.write(str(num) + '\n')
	f.close()
