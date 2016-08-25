from sklearn.datasets import load_svmlight_file
from IPython import embed
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

# Loading datasets with i = 100000
X_train,y_train = load_svmlight_file("../data/train_numeric.svm")
X_test,y_test = load_svmlight_file("../data/test_numeric.svm")

# Defining pipeline and params
lr = LogisticRegression()
pipeline = Pipeline(steps=[('lr', lr)])
params = {
	# "lr__C" = []
}
cv = GridSearchCV(pipeline,params)

# Fitting  CV
cv.fit(X_train[:,0:-1],y_train)
best_model = cv.best_estimator_


#Predicting and evaluating
y_pred = best_model.predict(X_test)
cm = confusion_matrix(map(lambda x: int(x),y_test),map(lambda x: int(x),y_pred))
print("Confusion Matrix")
print(cm)
print("MMC")
print(matthews_corrcoef(map(lambda x: int(x),y_test),map(lambda x: int(x),y_pred)))


