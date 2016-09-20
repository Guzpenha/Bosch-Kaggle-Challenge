from IPython import embed
import numpy as np

import csv
from scipy import sparse
import itertools
import re

from sklearn.utils import check_random_state, check_array
from sklearn.base import TransformerMixin

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


def load_labels(file_name):
	file = open(file_name)
	csvreader = csv.reader(file)
	# ignore headers
	next(csvreader, None)
	labels = []
	for line in csvreader:
	    d = np.array(line, dtype=str)
	    labels.append(int(d[-1]))
	return np.array(labels)

class TimeRangeClosestLabelsPercentage(TransformerMixin):
	def __init__(self,file_name):
		file = open(file_name)
		csvreader = csv.reader(file)
		headers = next(csvreader, None)
		self.dim = len(headers)
		self.no_suffix = np.array(["Id"] + [head.split("_")[0]+"_"+head.split("_")[1] for head in headers if head !="Id"])
	def fit(self, X, y):
		self.buckets_counts	= {}
		for index,col,v in itertools.izip(X.row, X.col, X.data):
			feature = str(v).split(".")[0]
			group = self.no_suffix[col]
			if group in self.buckets_counts:				
				if feature in self.buckets_counts[group]:
					self.buckets_counts[group][feature][y[index]] += 1
				else:
					self.buckets_counts[group][feature] = [0,0] 
					self.buckets_counts[group][feature][y[index]] += 1	
			else:
				self.buckets_counts[group] = {}
				self.buckets_counts[group][feature] = [0,0] 
				self.buckets_counts[group][feature][y[index]] += 1			
		return self

	def transform(self, X,closeness = 3):
		new_data = []
		new_col = []
		new_row = []
		for index,col,v in itertools.izip(X.row, X.col, X.data):
			feature = str(v).split(".")[0]
			total_0 = 0
			total_1 = 1
			for close_time in range(int(feature)-closeness,int(feature)+closeness):
				if str(close_time) in self.buckets_counts[self.no_suffix[col]]:
					total_0 += self.buckets_counts[self.no_suffix[col]][str(close_time)][0]
					total_1 += self.buckets_counts[self.no_suffix[col]][str(close_time)][1]
			new_data.append(float(total_1)/(total_1+total_0))
			new_row.append(index)
			new_col.append(col)									

		result = sparse.csr_matrix((new_data, (new_row, new_col)), (max(X.row)+1, self.dim))
		return result

# def load_date_closest_labels_percentage(file_name, labels_file, closeness = 3):
# 	file = open(file_name)
# 	csvreader = csv.reader(file)
# 	headers = next(csvreader, None)
# 	i=0
# 	buckets_counts = {}	
# 	no_suffix = np.array(["Id"] + [head.split("_")[0]+"_"+head.split("_")[1] for head in headers if head !="Id"])

# 	labels = load_labels(labels_file)
# 	print("labels loaded")
# 	#embed()
# 	for line in csvreader:
# 		d = np.array(line, dtype=str)
# 		c = np.where(d != '')[0]
# 		# print(labels[i])
# 		# print(d[c[1:]])
# 		# print(no_suffix[c[1:]])
# 		visited_groups_features = {}	

# 		for index in c[1:]:
# 			feature = d[index].split(".")[0]
# 			group = no_suffix[index]
# 			if((group,feature) not in visited_groups_features):
# 				if group in buckets_counts:				
# 					if feature in buckets_counts[group]:
# 						buckets_counts[group][feature][labels[i]] += 1
# 					else:
# 						buckets_counts[group][feature] = [0,0] 
# 						buckets_counts[group][feature][labels[i]] += 1	
# 				else:
# 					buckets_counts[group] = {}
# 					buckets_counts[group][feature] = [0,0] 
# 					buckets_counts[group][feature][labels[i]] += 1
# 				visited_groups_features[(group,feature)] = True
# 		i+=1
# 		# print(buckets_counts)
# 		# if (i%5==0):
# 			# break

# 	print("buckets counted")
# 	#embed()
# 	# Second pass in X in order to create features from buckets_counts
# 	file = open(file_name)
# 	csvreader = csv.reader(file)
# 	headers = next(csvreader, None)
# 	i=0
# 	values = []
# 	for line in csvreader:
# 		d = np.array(line, dtype=str)
# 		c = np.where(d != '')[0]
# 		visited_groups_features = {}
# 		total = [0,0]
# 		for index in c[1:]:
# 			feature = d[index].split(".")[0]
# 			if((no_suffix[index],feature) not in visited_groups_features):				
# 				visited_groups_features[(no_suffix[index],feature)] = True
# 				for close_time in range(int(feature)-closeness,int(feature)+closeness):					
# 					# if(labels[i]==1):
# 					# 	print(close_time)
# 					if str(close_time) in buckets_counts[no_suffix[index]]:
# 						total[0]+=buckets_counts[no_suffix[index]][str(close_time)][0]
# 						total[1]+=buckets_counts[no_suffix[index]][str(close_time)][1]
# 				# if(labels[i]==1):
# 					# print(labels[i])
# 					# print(buckets_counts[no_suffix[index]][feature])
# 		# if(labels[i]==1):
# 		# print(str(total) + " "+ str(labels[i]))
# 		values.append([total[1]/float(total[0]+total[1]), labels[i]]if (total[0]+total[1])!=0 else None)
# 		i+=1
# 	print("second pass made")
# 	#embed()
# 	sum_0 = reduce(lambda x, y: x+y ,[v[0] for v in values if v!=None and v[1]==0])
# 	sum_1 = reduce(lambda x, y: x+y ,[v[0] for v in values if v!=None and v[1]==1])
# 	print("avg 0: {}".format(sum_0/float(len([v[0] for v in values if v!=None and v[1]==0]))))
# 	print("avg 1: {}".format(sum_1/float(len([v[0] for v in values if v!=None and v[1]==1]))))
# 	return np.array(map(lambda x: [x[0]] if x else [0], values))

def load_time_spent_by_station_features(file_name, batch = 100000):
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
		for non_missing_index in c[1:]:
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
			#break
			
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
		# Force sparse format to be csc
		X = check_array(X, accept_sparse="csc")
		n_samples, n_features = X.shape
		# count the number of possible categorical values per feature
		self.categories = [np.unique(X.data[X.indptr[j]:X.indptr[j+1]]) for j in np.arange(n_features)]
		return self
	def transform(self, X):
		# Force sparse format to be csc
		X = check_array(X, accept_sparse="csc")
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
	X = sparse.csr_matrix([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2], [0, 1, 3]])
	enc.fit(X)

	print(X.toarray())

	X_t = enc.transform(sparse.csc_matrix([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 4], [0, 1, 3]]))
	print(X_t.toarray())

	print(X.nnz)
	print(X_t.nnz)
