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
			data.append(d[c[1:]].astype(float))
		else:
			dim = d.shape[0] - 2
			rows.append(np.repeat(row, c.shape[0] - 2))
			columns.append(c[1:-1] - 1)
			data.append(d[c[1:-1]].astype(float))	
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


	return matrix, np.asarray(label)

def predict_batch(estimator, file_name, batch=10000):
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
	pred = np.ndarray(0)
	for line in csvreader:
		d = np.array(line, dtype=str)
		
		c = np.where(d != '')[0]
		
		dim = d.shape[0] - 1
		rows.append(np.repeat(row, c.shape[0] - 1))
		columns.append(c[1:] - 1)
		data.append(d[c[1:]].astype(float))

		row = row + 1

		if(row%batch == 0):
			rows = np.concatenate(rows)
			
			#columns.append(scol)
			columns = np.concatenate(columns)
			
			#data.append(sdata)
			data = np.concatenate(data)


			matrix = sparse.csr_matrix((data, (rows, columns)), (batch, dim))
			
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

		matrix = sparse.csr_matrix((data, (rows, columns)), (row, dim))

		# cleaning data 
		rows = []
		columns = []
		data = []

		pred = np.concatenate((pred, estimator.predict(matrix)))

	return pred