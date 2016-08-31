import numpy as np

import csv
from scipy import sparse

def tozero(a):
	return 0 if a == '' else float(a) 

def load_dataset(file_name, batch = 1000, no_label = False):
	file = open(file_name)

	csvreader = csv.reader(file)
	# ignore headers
	next(csvreader, None)
	
	rows = []
	columns = []
	data = []
	row = 0

	srow = np.ndarray(0)
	scol = np.ndarray(0)
	sdata = np.ndarray(0)

	matrix = None
	label = []
	i = 0
	for line in csvreader:
		d = np.array(line, dtype=str)
		
		c = np.where(d != '')[0]
		
		if no_label:
			dim = d.shape[0] - 1
			rows.append(np.repeat(row, c.shape[0] - 1))
			columns.append(c[1:] - 1)
			data.append(d[c[1:]].astype(np.float64))
		else:
			dim = d.shape[0] - 2
			rows.append(np.repeat(row, c.shape[0] - 2))
			columns.append(c[1:-1] - 1)
			data.append(d[c[1:-1]].astype(np.float64))	
			label.append(int(d[-1]))	

		row = row + 1

		if(row%batch == 0):
			rows = np.concatenate(rows)
			
			#columns.append(scol)
			columns = np.concatenate(columns)
			
			#data.append(sdata)
			data = np.concatenate(data)

			if matrix is None:
				matrix = sparse.csr_matrix((data, (rows, columns)), (batch, dim))
			else:
				matrix = sparse.vstack((matrix, sparse.csr_matrix((data, (rows, columns)), (batch, dim))))
			
			# cleaning data 
			rows = []
			columns = []
			data = []
			
			print(i)
			row = 0
			i = i + 1

	if(not row == 0):
		rows = np.concatenate(rows)
		columns = np.concatenate(columns)
		data = np.concatenate(data)

		if matrix is None:
			matrix = sparse.csr_matrix((data, (rows, columns)), (row, dim))
		else:
			matrix = sparse.vstack((matrix, sparse.csr_matrix((data, (rows, columns)), (row, dim))))

		# cleaning data 
		rows = []
		columns = []
		data = []

	if no_label:
		return matrix


	return matrix, np.asarray(label, dtype=int)

def predict_batch(estimator, file_name, batch=10000):
	import collections
	readers = []
	if isinstance(file_name, str):
		readers = [csv.reader(open(file_name))]
	elif isinstance(file_name, collections.Iterable):
		readers = [csv.reader(open(f)) for f in file_name]
	else:
		return 0
	for reader in readers:
		# ignore headers
		next(reader, None)
	rows = []
	columns = []
	data = []
	row = 0
	matrix = None
	label = []
	i = 0
	pred = np.ndarray(0)
	for line in readers[0]:	
		d = np.array(line[1:], dtype=str)
		for reader in readers[1:]:
			line = reader.next()
			d = np.concatenate((d, np.array(line[1:], dtype=str)))
		c = np.where(d != '')[0]
		dim = d.shape[0]
		rows.append(np.repeat(row, c.shape[0]))
		columns.append(c)
		data.append(d[c].astype(np.float16))
		row = row + 1
		if(row%batch == 0):
			rows = np.concatenate(rows)	
			#columns.append(scol)
			columns = np.concatenate(columns)		
			#data.append(sdata)
			data = np.concatenate(data)
			matrix = sparse.csr_matrix((data, (rows, columns)), (batch, dim)).toarray()	
			# cleaning data 
			rows = []
			columns = []
			data = []
			pred = np.concatenate((pred, estimator.predict(matrix)))
			print(i)
			row = 0
			i = i + 1
	if(not row == 0):
		rows = np.concatenate(rows)
		columns = np.concatenate(columns)
		data = np.concatenate(data)
		matrix = sparse.csr_matrix((data, (rows, columns)), (row, dim)).toarray()
		# cleaning data 
		rows = []
		columns = []
		data = []
		pred = np.concatenate((pred, estimator.predict(matrix)))
	return pred

#pred = predict_batch(rf, ['../data/test_numeric.csv', '../data/test_date.csv'], batch = 1000)


from sklearn.base import TransformerMixin

'''
def onehot(X):
	n_samples, n_features = X.shape
	
	# count the number of passible categorical values per feature
	categories = [np.unique(X.data[X.indptr[j]:X.indptr[j+1]]) for j in np.arange(n_features)]
	
	# empty matrix
	result = sparse.lil_matrix((n_samples, sum(d)))
	n_features_sofar = 0
	for j in np.arange(n_features):
		# get values of the j-th colunm
		col = X.data[X.indptr[j]:X.indptr[j+1]]
		# j-th colunm's possible values
		cats = np.unique(col)
		
		# find the matches
		cols = np.where(col == cats[:, np.newaxis])[0]
	
		# insert into the sparse matrix
		result[X.indices[X.indptr[j]:X.indptr[j+1]], j + n_features_sofar + cols] = 1
		
		n_features_sofar = n_features_sofar + cats.shape[0] - 1

	# converto to csr format and return		
	return result.tocsr()
'''

class SparseOneHotEncoder(TransformerMixin):
	def __init__(self):
		super(SparseOneHotEnconder, self).__init__()
	def fit(self, X, y=None):
		n_samples, n_features = X.shape
		# count the number of passible categorical values per feature
		self.categories = [np.unique(X.data[X.indptr[j]:X.indptr[j+1]]) for j in np.arange(n_features)]
		return self
	def transform(self, X):
		n_samples, n_features = X.shape
		count_unique_cats = [cats.shape[0] for cats in self.categories]
		n_new_features = sum(count_unique_cats)
		# empty matrix
		result = sparse.lil_matrix((n_samples, n_new_features))
		n_features_sofar = 0
		for j, cats in enumerate(self.categories):
			# find the matches
			cols = np.where(X.data[X.indptr[j]:X.indptr[j+1]] == cats[:, np.newaxis])
			if len(cols) == 2:
				# sort by column ids
				cols = cols[0][np.argsort(cols[1])]
			else:
				cols = cols[0]
			# insert into the sparse matrix
			result[X.indices[X.indptr[j]:X.indptr[j+1]], j + n_features_sofar + cols] = 1
			n_features_sofar = n_features_sofar + cats.shape[0] - 1
		# convert to csr format and return		
		return result.tocsr()

if __name__ == '__main__':
	enc = SparseOneHotEncoder()
	X = sparse.csc_matrix([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2], [0, 1, 3]])
	enc.fit(X)


	print(X.toarray())
	X_t = enc.transform(X)
	print(X_t.toarray())

	print(X.nnz)
	print(X_t.nnz)