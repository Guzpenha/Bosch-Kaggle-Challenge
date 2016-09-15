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
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
import xgboost as xgb

from scipy.sparse import csr_matrix, hstack
from scipy.io import mmread, mmwrite
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
	X_train = scipy.sparse.hstack((X_train,X_date)).tocsr()
	return X_train, y_train

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
	X, y = load_full_dataset()
	# embed()

	# Dimensionality reduction
	# t_svd = TruncatedSVD(random_state=seed)	
	# gsr = GaussianRandomProjection(random_state=seed)
	# spr = SparseRandomProjection(random_state=seed)
	
	# Estimators
	xboost = xgb.XGBClassifier(seed=0)
	# rf = RandomForestClassifier(n_estimators=300, max_features=0.08, n_jobs=-1, random_state=seed)
	
	# Defining pipeline and params
	# pipeline = Pipeline(steps=[('tsvd',t_svd),('xboost', xboost)])
	# pipeline = Pipeline(steps=[('spr',spr),('xboost', xboost)])
	pipeline = Pipeline(steps=[('xboost', xboost)])
	params = {
		#"rf__max_features" : ['log2', 'sqrt', 0.08, 0.15, 0.3, 0.7, 1.0]
		#"xboost__learning_rate" : [0.1, 0.3, 0.7, 1.0]
		# "xboost__max_depth": [5,8,10,12],
		# "xboost__min_child_weight": [1,3,6],
		# "xboost__n_estimators": [100,150,200]		
		# "spr__n_components": ['auto',1000,500],
		# "gsr__n_components": ['auto',1000,500],
		# "tsvd__n_components": [1000,500],
		"xboost__max_depth": [12],
		"xboost__min_child_weight": [3],
		"xboost__n_estimators": [200]
	}

	print("Spliting data into train and test sets")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
	del X, y

	#Fitting custom CV with correct ratios
	print("Running CV")
	#embed()
	best_score, best_params, ratio, idx = val.GridsearchBestRatio(X_train.tocsc(), y_train, pipeline, scoring=score_mcc, verbose=5, ratios=[0.05], params=params)	

	# set params and refit
	pipeline.set_params(**best_params)
	pipeline.fit(X_train[idx], y_train[idx])
	# evalutate it
	print(score_mcc(pipeline, X_test, y_test))

	# Predicting test data and saving it for submission
	if(args.make_predictions):
		# fit train + test positive instances
		pipeline.fit(scipy.sparse.vstack((X_train[idx], X_test[y_test == 1])).tocsc(), np.concatenate((y_train[idx], y_test[y_test==1])))

		# free memory
		del X_train, X_test

		X_board = pre.load_dataset("../data/test_numeric.csv", batch = 100000, no_label=True)
		X_board_cat = mmread('../data/test_categorical')
		X_board_date = pre.load_date_features("../data/test_date.csv", batch = 100000)

		csvreader = csv.reader(open("../data/train_numeric.csv"))
		header = next(csvreader, None)
		X_board = scipy.sparse.hstack((X_board, extract_missing(X_board, header[1:-1]), X_board_cat)).tocsr()
		X_board = scipy.sparse.hstack((X_board,X_board_date)).tocsr()
		
		del X_board_cat, X_board_date

		df = pd.read_csv("../data/sample_submission.csv")
		df['Response'] = pipeline.predict(X_board.tocsc())
		df.to_csv("../data/submission_%s.csv" % pipeline.steps[0][0], index=False)
