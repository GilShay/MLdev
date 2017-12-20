#doesn't work
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from PrepareCSVdata import load_data
from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


csv_path = os.path.join("datasets/housing", "housing.csv")
csv_data = load_data(csv_path)
target = csv_data["ocean_proximity"].copy()
# target = target.values.tolist()
# target = np.array(target)
# target.reshape(-1,1)
print target

# columns = csv_data.to_dict('split')
# columns = columns['columns']
# print columns

csv_data = csv_data.drop("ocean_proximity", axis=1)
# print csv_data
csv_data = csv_data.to_dict('records')
v = DictVectorizer(sparse=False)
X = v.fit_transform(csv_data)
print X

forest_reg = RandomForestClassifier(n_estimators=20)
# forest_reg.fit(X, target)
#


parameters = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

# svc = svm.SVC()
clf = RandomizedSearchCV(forest_reg, parameters, n_iter=20)
clf.fit(X, target)
# print clf.best_params_


