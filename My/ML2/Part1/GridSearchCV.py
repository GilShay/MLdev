from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from PrepareCSVdata import load_data
from sklearn import svm
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.ensemble import RandomForestRegressor


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

forest_reg = RandomForestRegressor()
# forest_reg.fit(X, target)
#


parameters = [{'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
# svc = svm.SVC()
clf = GridSearchCV(forest_reg, parameters, cv=5, scoring='neg_mean_squared_error')
clf.fit(X, target)
# print clf.best_params_

































# forest_reg.predict(data_to_check)

# # # columns.reshape(-1,1)
#
# data = csv_data['data']
# # print data
# # print columns
# # data = np.matrix(data)
# # # data.reshape(-1,1)
# # print data
#

# columns.pivot()

# csv_data["income_cat"] = np.ceil(csv_data["median_income"]/1.5)
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(csv_data, csv_data["income_cat"]):
#     train_set = csv_data.loc[train_index]
#     test_set = csv_data.loc[test_index]
# # print train_set.shape, test_set.shape
#
# housing_labels = train_set["median_house_value"].copy()
# print housing_labels
#