from IPython import embedfrom optparse import OptionParser

from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix, hstack
from scipy.io import mmread
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np

import preprocessing as pre

import xgboost as xgb

from optparse import OptionParser

from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from scipy.sparse import csr_matrix, hstack 
from scipy.io import mmread
from sklearn.cross_validation import train_test_split

import pandas as pd
import numpy as np

import preprocessing as pre

import xgboost as xgb

import csv
from scipy import sparse

def score_mcc(estimator, X, y):
	return matthews_corrcoef(estimator.predict(X), y)

if __name__ == "__main__":

	parser = OptionParser()
	parser.add_option("-p","--make_predictions", help="make predictions to file.",default = False)
	args = parser.parse_args()[0]

	seed = 42
	np.random.seed(seed)

	# Loading datasets with i = 100000
	X_train, y_train = pre.load_dataset("../data/train_numeric.csv", batch = 100000)

	X_train_cat = scipy.io.mmread('../data/train_categorical')


	csvreader = csv.reader(open("../data/train_numeric.csv"))
	header = next(csvreader, None)

	X_train = scipy.sparse.hstack((X_train, extract_missing(X_train, header[1:-1]), X_train_cat)).tocsr()

	X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)

	y0_idx = np.where(y_train == 0)[0]
	y1_idx = np.where(y_train == 1)[0]
	np.random.shuffle(y0_idx)

	idx = np.concatenate((y1_idx, y0_idx[:(20*y1_idx.shape[0])]))

	# I'm doing this in order to save memory (the ideal it is not that hehe)
	X_train, y_train = X_train[idx], y_train[idx]
	
	# Defining pipeline and params
	rf = RandomForestClassifier(n_estimators=300, max_features=0.08, n_jobs=-1, random_state=seed)

	xboost = xgb.XGBClassifier(max_depth=100, n_estimators=200, seed=0)

	pipeline = Pipeline(steps=[('xboost', xboost)])
	params = {
		#"rf__max_features" : ['log2', 'sqrt', 0.08, 0.15, 0.3, 0.7, 1.0]
		#"boost__learning_rate" : [0.1, 0.3, 0.7, 1.0]
	}
	cv = GridSearchCV(pipeline, params, scoring=score_mcc, verbose=5)

	# Fitting  CV
	cv.fit(X_train, y_train)
	best_model = cv.best_estimator_

	print(score_mcc(cv, X_test, y_test))


	best_model.fit(scipy.sparse.vstack((X_train, X_test)), np.concatenate((y_train, y_test)))

	# Predicting test data and saving it for submission
	if(args.make_predictions):
		X_board = pre.load_dataset("../data/test_numeric.csv", batch = 100000, no_label=True)
		X_board_cat = scipy.io.mmread('../data/test_categorical')
		X_board = scipy.sparse.hstack((X_board, extract_missing(X_board, header[1:-1]), X_board_cat))
		
		df = pd.read_csv("../data/sample_submission.csv")
		df['Response'] = best_model.predict(X_board)
		df.to_csv("../data/submission_%s.csv" % pipeline.steps[0][0], index=False)
