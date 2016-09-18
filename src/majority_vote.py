from os import listdir
from os.path import isfile, join

import pandas as pd
import numpy as np

from optparse import OptionParser

def majority_vote(mypath):
	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	votes = None

	for file in onlyfiles:
		pred = pd.read_csv(mypath + file)['Response'].as_matrix()
		n_samples = pred.shape[0]

		if votes is None:
			votes = np.zeros((n_samples, np.unique(pred).shape[0]))

		votes[np.arange(n_samples, dtype=int), pred] += 1

	return np.argmax(votes, axis=1)

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("-f","--folder", help="folder with the submission files", default="../data/best_submissions/")
	args = parser.parse_args()[0]

	submission = pd.read_csv("../data/sample_submission.csv")
	submission['Response'] = majority_vote(args.folder)
	submission.to_csv("../data/submission_majority2.csv", index=False, dtype=int)