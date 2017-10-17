# Import libraries:
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import model_selection, metrics  # Additional scklearn functions
from sklearn.model_selection import GridSearchCV  # Perforing grid search
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from sklearn import preprocessing



# rcParams['figure.figsize'] = 10, 5



train = pd.read_csv('ayur_all_users.csv', header=0, encoding='latin1')
train.drop(['id', 'Unnamed: 0', 'index', 'conv_rate'], axis=1, inplace=True)
train['bounce_rate'] = train['bounce_rate'].astype(float)
# train['conv_rate'] = train['conv_rate'].astype(float)


x = train.values
x_scaled = preprocessing.MinMaxScaler().fit_transform(x)
df = pd.DataFrame(x_scaled, columns=train.columns)
train, test = train_test_split(df, test_size = 0.4)

target = 'with_conversion'
IDcol = 'id'


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
		xgtest = xgb.DMatrix(dtest[predictors].values)
		cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
						  metrics='auc', early_stopping_rounds=early_stopping_rounds)
		alg.set_params(n_estimators=cvresult.shape[0])
	
	# Fit the algorithm on the data
	alg.fit(dtrain[predictors], dtrain['with_conversion'], eval_metric='auc')
	
	# Predict training set:
	dtrain_predictions = alg.predict(dtrain[predictors])
	dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]
	dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]
	
	# Print model report:
	print("\nModel Report")
	print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['with_conversion'].values, dtrain_predictions))
	print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['with_conversion'], dtrain_predprob))
	
	#     Predict on testing data:
	dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
	print('AUC Score (Test): %f' % metrics.roc_auc_score(test['with_conversion'], dtest_predprob))
	
	
	feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
	feat_imp.plot(kind='bar', title='Feature Importances')
	plt.ylabel('Feature Importance Score')
	plt.tight_layout()
	#plt.show()
	plt.savefig('forsolving.com/result.png')

	return {"accuracy": "%.4g" % metrics.accuracy_score(dtrain['with_conversion'].values, dtrain_predictions),
			"train": "%f" % metrics.roc_auc_score(dtrain['with_conversion'], dtrain_predprob),
			"test": "%f" % metrics.roc_auc_score(test['with_conversion'], dtest_predprob)
			}
	
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
# modelfit(xgb1, train, predictors)

if __name__ == "__main__":
	def param_test1():
		param_test1 = {
		 'max_depth':range(3,10,2),
		 'min_child_weight':range(1,6,2)
		}
		gsearch1 = GridSearchCV(estimator = XGBClassifier(
			learning_rate =0.1,
			n_estimators=140,
			max_depth=5,
			min_child_weight=1,
			gamma=0,
			subsample=0.8,
			colsample_bytree=0.8,
			objective= 'binary:logistic',
			nthread=4,
			scale_pos_weight=1, seed=27),
			param_grid = param_test1,
			scoring='roc_auc',
			n_jobs=4,
			iid=False,
			cv=5)
		gsearch1.fit(train[predictors],train[target])
		print(gsearch1.grid_scores_,'\n', 'best params', gsearch1.best_params_,'\n', 'best score', gsearch1.best_score_)
	
	def param_test2():
		param_test2 = {
			'max_depth': [2, 3, 4],
			'min_child_weight': [2, 1, 3]
		}
		gsearch2 = GridSearchCV(
			estimator=XGBClassifier(learning_rate=0.1,
			n_estimators=140,
			max_depth=3,
			min_child_weight=2,
			gamma=0,
			subsample=0.8,
			colsample_bytree=0.8,
			objective='binary:logistic',
			nthread=4,
			scale_pos_weight=1,
			seed=27),
			param_grid=param_test2,
			scoring='roc_auc',
			n_jobs=4,
			iid=False,
			cv=5)
		gsearch2.fit(train[predictors], train[target])
		print(gsearch2.grid_scores_,'\n', 'best params', gsearch2.best_params_,'\n', 'best score', gsearch2.best_score_)
		
	def param_test2b():
		param_test2b = {
			'min_child_weight': [1,2,4, 6, 8, 10, 12]
		}
		gsearch2b = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=4,
														 min_child_weight=2, gamma=0, subsample=0.8,
														 colsample_bytree=0.8,
														 objective='binary:logistic', nthread=4, scale_pos_weight=1,
														 seed=27),
								 param_grid=param_test2b, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch2b.fit(train[predictors], train[target])
		modelfit(gsearch2b.best_estimator_, train, predictors)
		print(gsearch2b.grid_scores_,'\n', 'best params', gsearch2b.best_params_,'\n', 'best score', gsearch2b.best_score_)
	
	def param_test3():
		param_test3 = {
			'gamma': [i / 10.0 for i in range(0, 5)]
		}
		gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=1,
														min_child_weight=1, gamma=0, subsample=0.8,
														colsample_bytree=0.8,
														objective='binary:logistic', nthread=4, scale_pos_weight=1,
														seed=27),
								param_grid=param_test3, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch3.fit(train[predictors], train[target])
		print(gsearch3.grid_scores_,'\n', 'best params', gsearch3.best_params_,'\n', 'best score', gsearch3.best_score_)
	
	
	xgb2 = XGBClassifier(
		learning_rate=0.1,
		n_estimators=1000,
		max_depth=2,
		min_child_weight=1,
		gamma=0,
		subsample=0.8,
		colsample_bytree=0.8,
		objective='binary:logistic',
		nthread=4,
		scale_pos_weight=1,
		seed=27)
	
	
	def param_test4():
		param_test4 = {
			'subsample': [i / 10.0 for i in range(6, 10)],
			'colsample_bytree': [i / 10.0 for i in range(6, 10)]
		}
		gsearch4 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=2,
														min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
														objective='binary:logistic', nthread=4, scale_pos_weight=1,
														seed=27),
								param_grid=param_test4, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch4.fit(train[predictors], train[target])
		print(gsearch4.grid_scores_,'\n', 'best params', gsearch4.best_params_,'\n', 'best score', gsearch4.best_score_)
	
	def param_test5():
		param_test5 = {
			'subsample': [i / 100.0 for i in range(75, 90, 5)],
			'colsample_bytree': [i / 100.0 for i in range(85, 100, 5)]
		}
		gsearch5 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=2,
														min_child_weight=1, gamma=0, subsample=0.8,
														colsample_bytree=0.8,
														objective='binary:logistic', nthread=4, scale_pos_weight=1,
														seed=27),
								param_grid=param_test5, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch5.fit(train[predictors], train[target])
		print(gsearch5.grid_scores_,'\n', 'best params', gsearch5.best_params_,'\n', 'best score', gsearch5.best_score_)

	def param_test6():
		param_test6 = {
			'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100]
		}
		gsearch6 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=2,
														min_child_weight=1, gamma=0, subsample=0.85,
														colsample_bytree=0.9,
														objective='binary:logistic', nthread=4, scale_pos_weight=1,
														seed=27),
								param_grid=param_test6, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch6.fit(train[predictors], train[target])
		print(gsearch6.grid_scores_,'\n', 'best params', gsearch6.best_params_,'\n', 'best score', gsearch6.best_score_)
	
	def param_test7():
		param_test7 = {
				'reg_alpha': [0, 0.0001, 0.00001, 0.001, 0.01]
			}
		gsearch7 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=177, max_depth=2,
															min_child_weight=1, gamma=0, subsample=0.85,
															colsample_bytree=0.9,
															objective='binary:logistic', nthread=4, scale_pos_weight=1,
															seed=27),
									param_grid=param_test7, scoring='roc_auc', n_jobs=4, iid=False, cv=5)
		gsearch7.fit(train[predictors], train[target])
		print(gsearch7.grid_scores_, '\n', 'best params', gsearch7.best_params_, '\n', 'best score', gsearch7.best_score_)
	
	
	xgb3 = XGBClassifier(
		learning_rate=0.1,
		n_estimators=1000,
		max_depth=2,
		min_child_weight=1,
		gamma=0,
		subsample=0.85,
		colsample_bytree=0.9,
		reg_alpha=0.0001,
		objective='binary:logistic',
		nthread=4,
		scale_pos_weight=1,
		seed=27)
	
	xgb4 = XGBClassifier(
		learning_rate=0.01,
		n_estimators=5000,
		max_depth=2,
		min_child_weight=1,
		gamma=0,
		subsample=0.85,
		colsample_bytree=0.9,
		reg_alpha=0.005,
		objective='binary:logistic',
		nthread=4,
		scale_pos_weight=1,
		seed=27)
	
	# param_test1()
	# param_test2()
	# param_test2b()
	# param_test3()
	# param_test4()
	# param_test5()
	# param_test6()
	# param_test7()
	
	
	
	#modelfit(xgb1, train, test, predictors)
	#modelfit(xgb2, train, test, predictors)
	#modelfit(xgb3, train, test, predictors)
	modelfit(xgb4, train, test, predictors)
