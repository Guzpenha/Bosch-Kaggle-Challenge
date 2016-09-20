import sys
sys.path.append('../')
import pandas as pd
import numpy as np
import csv
import preprocessing as pre
from IPython import embed
from scipy.sparse import csr_matrix, hstack 

# loading closest parts label
X_date_numeric = pre.load_dataset("../../data/train_date.csv", batch = 1000, no_label = True).tocoo()
x,y = pre.load_dataset("../../data/train_numeric.csv", batch = 1000)
clstLabels = pre.TimeRangeClosestLabelsPercentage("../../data/train_date.csv")
clstLabels.fit(X_date_numeric,y)
X_date_closest_y_ratio = clstLabels.transform(X_date_numeric,closeness = 3)
embed()
# print(X_date.shape)
# embed()

#Reading time spent in stations features
# X_date = pre.load_time_spent_by_station_features("../../data/train_date.csv", batch = 100000)
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

