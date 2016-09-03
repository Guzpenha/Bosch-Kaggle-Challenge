from IPython import embed
import numpy as np

import csv
from scipy import sparse

import re

def extract_missing(X, header):
	# removing suffix related to feature
	# we are assuming that once the part reaches a station in a given line
	# the pair <line,station> was used
	no_suffix = np.array([re.sub(r'\_F\d+', '', head) for head in header])
	
	# get the unique pair <line,station> and their occurance
	unique, counts = np.unique(no_suffix, return_counts=True)
	
	# np.unique function sorts by charecters order, but we also want
	# to keep the order of the station (e.g. S2 comes right after S1, if we
	# keep the np.unique order after S1 will come S10)
	# in order to achieve what we want we use a stable sort method such as merge
	# and sort by the size of the string
	order = np.argsort([len(s) for s in unique], kind='merge')
	
	# resorting accordly to the explanation above
	unique, counts = unique[order], counts[order]
	
	# getting the initial colunms of each pair <line,station>
	cols = np.concatenate(([0], np.cumsum(counts)[:-1]))
	
	# binary matrix of patterns
	pattern = X[:, cols]
	pattern.data = np.ones(pattern.data.shape)
	
	return pattern

def tozero(a):
	return 0 if a == '' else float(a)

def load_date_features(file_name, batch = 100000):
	file = open(file_name)
	csvreader = csv.reader(file)
	
	headers = next(csvreader, None)
	
	headers_mappings = {}
	headers_key_values = {}
	count_header_groups = 0
	for i,h in enumerate(headers):
		if(h!="Id"):
			group ='_'.join(h.split("_")[0:-1])			
			if(group in headers_mappings):
				headers_mappings[group][1] = i
			else:
				headers_key_values[group] = count_header_groups
				count_header_groups+=1
				headers_mappings[group] = [0,0]
				headers_mappings[group][0] = i
				headers_mappings[group][1] = i	
	# print(headers_mappings)
	rows = []
	columns = []
	data = []
	dim = len(headers_mappings)		
	matrix = None
	row = 0
	i = 0

	for line in csvreader:
		d = np.array(line, dtype=str)
		c = np.where(d != '')[0]
		done_ranges = {}
		for non_missing_index in c:
			if(non_missing_index != 0):
				group = '_'.join(headers[non_missing_index].split("_")[0:-1])
				if(group not in done_ranges):			
					group_features = d[headers_mappings[group][0]:headers_mappings[group][1]]
					features_values = [float(f) for f in group_features if f !=""]				
					if len(features_values) !=0:											
						# print("time-spent: " + str(max(features_values)-min(features_values)))
						done_ranges[group] = max(features_values)-min(features_values)
		# print(done_ranges)
		rows.append(np.repeat(row, len(done_ranges.keys())))
		# print(np.repeat(row, len(done_ranges.keys())))		
		columns.append(np.array([headers_key_values[x] for x in done_ranges.keys()]))
		# print(np.array([headers_key_values[x] for x in done_ranges.keys()]))
		data.append([done_ranges[g] for g in done_ranges.keys()])
		# print([done_ranges[g] for g in done_ranges.keys()])
		row+=1

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

	return matrix


def load_dataset(file_name, batch = 10000, no_label = False):
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
		super(SparseOneHotEncoder, self).__init__()
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
			rows = X.indices[X.indptr[j]:X.indptr[j+1]]
			if len(cols) == 2:
				# ignore rows which categorical value does not exist in training set
				rows = rows[np.sort(cols[1])]
				# sort by column ids
				cols = cols[0][np.argsort(cols[1])]
			else:
				cols = cols[0]
			# insert into the sparse matrix
			result[rows, j + n_features_sofar + cols] = 1
			n_features_sofar = n_features_sofar + cats.shape[0] - 1
		# convert to csr format and return		
		return result.tocsr()

if __name__ == '__main__':
	enc = SparseOneHotEncoder()
	X = sparse.csc_matrix([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2], [0, 1, 3]])
	enc.fit(X)

	print(X.toarray())

	X_t = enc.transform(sparse.csc_matrix([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 4], [0, 1, 3]]))
	print(X_t.toarray())

	print(X.nnz)
	print(X_t.nnz)