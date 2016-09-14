from IPython import embed
from optparse import OptionParser
import os  

from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

import xgboost as xgb

from scipy.sparse import csr_matrix, hstack
from scipy.io import mmread
from scipy.io import mmwrite
from scipy import sparse
import scipy

import pandas as pd
import numpy as np
import csv

from  preprocessing import extract_missing
import preprocessing as pre
import validation as val


def load_full_dataset():	
	# If X_train.mtx have not been generated yet 
	if( not os.path.exists('../data/X_train.mtx')):
		X_train, y_train = pre.load_dataset("../data/train_numeric.csv", batch = 100000)
		X_date = pre.load_date_features("../data/train_date.csv", batch = 100000)
		# TO DO
		# X_train_cat = 
		mmwrite('../data/X_train',X_train)		
		mmwrite('../data/train_date',X_date)
		# mmwrite('../data/train_categorical',X_train_cat)

	else:
		X_train = mmread('../data/X_train')
		X_train = X_train.tocsr()
		y_train = pre.load_labels("../data/train_numeric.csv")
		X_date = mmread('../data/train_date')
		X_train_cat = mmread('../data/train_categorical')

	csvreader = csv.reader(open("../data/train_numeric.csv"))
	header = next(csvreader, None)
	X_train = scipy.sparse.hstack((X_train, extract_missing(X_train, header[1:-1]), X_train_cat)).tocsr()
	X_train = scipy.sparse.hstack((X_train,X_date))	

	return X_train.tocsc(), y_train

def score_mcc(estimator, X, y):
	import sklearn as skl
	return skl.metrics.matthews_corrcoef(estimator.predict(X), y)

if __name__ == "__main__":

	parser = OptionParser()
	parser.add_option("-p","--make_predictions", help="make predictions to file.",default = False)
	args = parser.parse_args()[0]

	seed = 42
	np.random.seed(seed)

	# Loading dataset with all features
	print("Loading features")
	X_train, y_train = load_full_dataset()		
	#embed()

	# Defining pipeline and params
	xboost = xgb.XGBClassifier(seed=0)
	# rf = RandomForestClassifier(n_estimators=300, max_features=0.08, n_jobs=-1, random_state=seed)
	pipeline = Pipeline(steps=[('xboost', xboost)])
	params = {
		#"rf__max_features" : ['log2', 'sqrt', 0.08, 0.15, 0.3, 0.7, 1.0]
		#"xboost__learning_rate" : [0.1, 0.3, 0.7, 1.0]
		# "xboost__max_depth": [5,8,10,12],
		# "xboost__min_child_weight": [1,3,6],
		# "xboost__n_estimators": [100,150,200]
		"xboost__max_depth": [12],
		"xboost__min_child_weight": [3],
		"xboost__n_estimators": [200]
	}
	
	#Fitting custom CV with correct ratios
	print("Running CV")
	#embed()
	X_train_80,X_test_20, y_train_80, y_test20 = train_test_split(X_train,y_train,test_size = 0.2, random_state = seed)
	best_params = val.GridsearchBestRatio(X_train, y_train, pipeline, scoring=score_mcc, verbose=5, ratios=[0.05], params=params)	
	best_model = xgb.XGBClassifier(n_estimators = best_params[1]["xboost__n_estimators"], max_depth = best_params[1]["xboost__max_depth"],min_child_weight=best_params[1]["xboost__min_child_weight"])
	best_model.fit(X_train_80,y_train_80)
	print(score_mcc(best_model,X_test_20,y_test20))

	# Predicting test data and saving it for submission
	if(args.make_predictions):
		X_board = pre.load_dataset("../data/test_numeric.csv", batch = 100000, no_label=True)
		X_board_cat = mmread('../data/test_categorical')
		X_board_date = pre.load_date_features("../data/test_date.csv", batch = 100000)		
		X_board = scipy.sparse.hstack((X_board, extract_missing(X_board, header[1:-1]), X_board_cat)).tocsr()
		X_board = scipy.sparse.hstack((X_board,X_board_date))
		
		del X_board_cat
		del X_board_date

		df = pd.read_csv("../data/sample_submission.csv")
		df['Response'] = best_model.predict(X_board)
		df.to_csv("../data/submission_%s.csv" % pipeline.steps[0][0], index=False)
