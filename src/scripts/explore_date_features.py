import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import csv
import preprocessing as pre
from IPython import embed
from scipy.sparse import csr_matrix, hstack 

# close parts label
X_date = pre.load_date_closest_labels_percentage("../../data/train_date.csv","../../data/train_numeric.csv")
# print(X_date.shape)
# embed()

#Reading date features

# X_date = pre.load_date_features("../data/train_date.csv", batch = 100000)
# print(X_date.shape)

# # Reading labels
# file = open("../data/train_numeric.csv")
# csvreader = csv.reader(file)
# # ignore headers
# next(csvreader, None)
# labels = []
# for line in csvreader:
#     d = np.array(line, dtype=str)
#     labels.append(int(d[-1]))
# print(len(labels))

# print("Cheking time spent in each groups")
# X_date[[i for i,l in enumerate(labels) if l == 0]].mean()
# X_date[[i for i,l in enumerate(labels) if l == 1]].mean()

# X_date[[i for i,l in enumerate(labels) if l == 0]].max()
# X_date[[i for i,l in enumerate(labels) if l == 1]].max()

