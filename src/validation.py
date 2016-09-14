from IPython import embed
import numpy as np

import csv
from scipy import sparse
import operator
import re

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from sklearn.utils import check_random_state

def SplitCV_OriRate(X, y, n_folds=3, shuffle=False, random_state=None, ratio=0.2):
	random_instance = check_random_state(random_state)
	y0_idx = np.where(y == 0)[0]
	y1_idx = np.where(y == 1)[0]
	random_instance.shuffle(y0_idx)
	idx = np.concatenate((y1_idx, y0_idx[:int(y1_idx.shape[0]/ratio)]))
	folds = StratifiedKFold(y[idx], n_folds=n_folds, shuffle=shuffle, random_state=random_state)
	ori_ratio = (y1_idx.shape[0]/float(y0_idx.shape[0]))
	lista = []
	for train_index, test_index in folds:
		n_zeros = (y[idx[test_index]] == 0).sum()
		n_ones = (y[idx[test_index]] == 1).sum()
		n_sel = (n_ones/ori_ratio) - n_zeros
		train_index = idx[train_index]
		test_index = np.concatenate((idx[test_index],
			random_instance.choice(y0_idx[int(y1_idx.shape[0]/ratio):], size=n_sel, replace=False)))
		lista.append((train_index, test_index))
	
	return lista, idx


def GridsearchBestRatio(X, y, estimator, n_iter=1, n_folds=3, fit_params={}, params={}, scoring=None, ratios=[0.2, 0.05], verbose=0):
	best_results=[]
	for ratio in ratios:
		best_score_avg = 0
		for i in range(int(n_iter)):
			splits, idx = SplitCV_OriRate(X, y, n_folds, ratio=ratio, random_state=32*i)
			cv = GridSearchCV(estimator, params, cv=splits, fit_params=fit_params, scoring=scoring, verbose=verbose, refit=False)
			cv.fit(X, y)
			best_score_avg = best_score_avg + cv.best_score_
		best_results.append((best_score_avg/float(n_iter), cv.best_params_, ratio))
	return sorted(best_results, key=operator.itemgetter(0), reverse=True)[0] + idx
