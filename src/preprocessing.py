import numpy as np

import csv
from scipy import sparse

def tozero(a):
	return 0 if a == '' else float(a) 

#rows, columns = 10, 100

file = open('datasets/train_numeric.csv')

csvreader = csv.reader(file)
next(csvreader, None)
rows = []
columns = []
data = []
row = 0

matrix = []
for line in csvreader:
	d = np.array(line, dtype=str)
	c = np.where(d != '')[0]
	#matrix.append(d[c].astype(float))
	rows.append(np.repeat(row, c.shape[0] - 1))
	columns.append(c[1:] - 1)
	data.append(d[c[1:]].astype(float))
	
	row = row + 1
	if(row%100000 == 0):
		break

data = np.concatenate(data)
rows = np.concatenate(rows)
columns = np.concatenate(columns)
matrix = sparse.coo_matrix((data, (rows, columns))).tocsc()

data = []
rows = []
columns = []

from sklearn.datasets import dump_svmlight_file

dump_svmlight_file(matrix[:,:-1], matrix[:,-1].toarray().ravel(), "datasets/train_numeric.svm")