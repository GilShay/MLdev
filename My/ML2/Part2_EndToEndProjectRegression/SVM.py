#SupportVectorMachine
from sklearn.svm import SVR
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(housing_path):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


csv_data = load_data("datasets/housing")
csv_data = csv_data[pd.notnull(csv_data['total_bedrooms'])]
# csv_data = pd.DataFrame(csv_data)
# print csv_data
# csv_data.to_csv("datasets/housing/try.csv")
encoder = LabelEncoder()
housing_cat = csv_data["ocean_proximity"]
housing_cat = encoder.fit_transform(housing_cat)
csv_data= csv_data.drop('ocean_proximity', 1)
csv_data.insert(loc=0, column='ocean_proximity', value=housing_cat)
csv_data.to_csv("datasets/housing/try.csv")

# print csv_data
svr = SVR(kernel='linear')
y = (csv_data["median_house_value"])
X = (csv_data["median_income"], csv_data["housing_median_age"],csv_data["households"], csv_data["population"], csv_data["longitude"], csv_data["latitude"] ,csv_data["total_rooms"], csv_data["ocean_proximity"], csv_data["total_bedrooms"])
X = np.asarray(X)
X = X.reshape(-1, 9)
# print X
y = np.asarray(y)
# print type(X)
# print type(y)

svr.fit(X, y)
user_median_income = raw_input("Please insert your income, and I'll print my estimation for your house value")
Info_to_check = [user_median_income, 41, 126, 322, -122.23, 37.88, 880, 3, 129]
Info_to_check = np.asarray(Info_to_check)
Info_to_check = Info_to_check.reshape(-1, 9)
print svr.predict(Info_to_check)



# n_samples, n_features = 10, 5
# np.random.seed(0)
# y = np.random.randn(n_samples)
# X = np.random.randn(n_samples, n_features)
# print type(X)
# print type(y)
# clf = SVR(C=1.0, epsilon=0.2)
# clf.fit(X, y)



# df = pd.DataFrame([[np.nan, 2, np.nan, 0], [3, 4, np.nan, 1],
#                    [np.nan, np.nan, 4, 5]],
#                  columns=list('ABCD'))
#
# print df
# df = df[pd.notnull(df['C'])]
# print df