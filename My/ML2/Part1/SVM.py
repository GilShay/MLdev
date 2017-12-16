#SupportVectorMachine
from sklearn.svm import SVR
import os
import pandas as pd
import numpy as np


def load_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#
csv_data = load_data("datasets/housing")
# print csv_data
svr = SVR(kernel='linear')
y = (csv_data["median_house_value"])
X = (csv_data["median_income"])
X = np.asarray(X)
X = X.reshape(-1, 1)
# print X
y = np.asarray(y)
# print type(X)
# print type(y)

svr.fit(X, y)
user_median_income = raw_input("Please insert your income, and I'll print my estimation for your house value")

print svr.predict(user_median_income)



# n_samples, n_features = 10, 5
# np.random.seed(0)
# y = np.random.randn(n_samples)
# X = np.random.randn(n_samples, n_features)
# print type(X)
# print type(y)
# clf = SVR(C=1.0, epsilon=0.2)
# clf.fit(X, y)

