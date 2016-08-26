from sklearn.datasets import load_svmlight_file
from IPython import embed
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

import pandas as pd

import preprocessing as pre

def score_mmc(estimator, X, y):
	return matthews_corrcoef(estimator.predict(X), y)

seed = 1234
np.random.seed(seed)

# Loading datasets with i = 100000
X_train, y_train = pre.load_dataset("../data/train_numeric.csv", batch = 100000)

y0_idx = np.where(y_train == 0)[0]
y1_idx = np.where(y_train == 1)[0]
np.random.shuffle(y0_idx)

idx = np.concatenate((y1_idx, y0_idx[:(10*y1_idx.shape[0])]))

# I'm doing this in order to save memory (the ideal it is not that hehe)
X_train, y_train = X_train[idx], y_train[idx]

#X_test,y_test = load_svmlight_file("../data/test_numeric.svm")

# Defining pipeline and params
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight="balanced_subsample", random_state=seed)
pipeline = Pipeline(steps=[('rf', rf)])
params = {
	# "lr__C" = []
}
cv = GridSearchCV(pipeline, params, scoring=score_mmc, verbose=5)

# Fitting  CV
cv.fit(X_train, y_train)
best_model = cv.best_estimator_

# saving memory again... 
X_train = pre.load_dataset("../data/test_numeric.csv", batch = 100000, no_label = True)

# Predicting test data and saving it for submission
df = pd.read_csv("../data/sample_submission.csv")
df['Response'] = best_model.predict(X_train)
df.to_csv("../data/submission_%s.csv" % pipeline.steps[0][0], index=False)
