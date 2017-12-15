import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# import pandas.plotting.sc



DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml/tree/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    # urllib.request.urlretrieve(housing_url, tgz_path)
    print tgz_path
    print housing_path
    housing_tgz = tarfile.open(tgz_path, "r")
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    return bytearray(hash(np.int64(identifier)).digest())[-1]<256*test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    # print in_test_set
    return data.loc[~in_test_set], data.loc[in_test_set]


csv_housing_data = load_housing_data()
print csv_housing_data
print csv_housing_data.info()
print csv_housing_data["ocean_proximity"].value_counts()
print csv_housing_data.describe()

# csv_housing_data.hist(bins=50, figsize=(20,15))
# plt.show()


train_set, test_set = split_train_test(csv_housing_data, 0.2)
print(len(train_set), "train +", len(test_set), "test")
housing_with_id = csv_housing_data.reset_index()
# print housing_with_id
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
print(len(train_set), "train +", len(test_set), "test")
housing_with_id["id"] = csv_housing_data["longitude"] * 1000 + csv_housing_data["latitude"]
train_set, test_set= split_train_test_by_id(housing_with_id, 0.2, "id")
print(len(train_set), "train +", len(test_set), "test")
train_set, test_set = train_test_split(csv_housing_data, test_size=0.2, random_state=42)
print(len(train_set), "train +", len(test_set), "test")


csv_housing_data["income_cat"] = np.ceil(csv_housing_data["median_income"]/1.5)
csv_housing_data["median_income"].hist()
csv_housing_data["income_cat"].hist()
# plt.show()
csv_housing_data["income_cat"].where(csv_housing_data["income_cat"]<5, 5.0, inplace=True)
csv_housing_data["median_income"].hist()
csv_housing_data["income_cat"].hist()
# plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(csv_housing_data, csv_housing_data["income_cat"]):
    start_train_set = csv_housing_data.loc[train_index]
    start_test_set = csv_housing_data.loc[test_index]
    # print train_index, test_index
    # print start_test_set, start_train_set
housing_1 = csv_housing_data["income_cat"].value_counts()/len(csv_housing_data)


# div = 0
# sum = 0
# for i in range(1, 5):
#     multi = housing_1[i]*i
#     sum += multi
#     div += i
# print sum/div

for set in (start_train_set, start_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

housing = start_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()

housing = start_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population",
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.show()

corr_matrix = housing.corr()
print corr_matrix


attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print corr_matrix


housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()
print housing_labels

print housing["total_bedrooms"]
housing.dropna(subset=["total_bedrooms"])
print housing["total_bedrooms"]


imputer =Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)


print imputer.statistics_
print housing_num.median().values
X = imputer.transform(housing_num)
print X
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
print housing_tr



encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
print housing_cat
print housing_cat_encoded
print encoder.classes_

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print housing_cat_1hot
print housing_cat_1hot.toarray()


encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
print housing_cat_1hot


rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/ X[:, household_ix]
        population_per_household = X[:, population_ix]/ X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/ X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
print housing.values
print housing_extra_attribs


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X[self.attribute_names].values


class LabelBinarizer_new(TransformerMixin, BaseEstimator):
    def fit(self, X, y = 0):
        self.encoder = None
        return self
    def transform(self, X, y = 0):
        if(self.encoder is None):
            print("Initializing encoder")
            self.encoder = LabelBinarizer();
            result = encoder.fit_transform(X)
        else:
            result = encoder.transform(X)
        return result;

num_attribs = list(housing_num)
print num_attribs
print "1"
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('arrtribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
   ])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print housing_num_tr
print housing_num_tr.shape
#


cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizer_new()),
    ])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])


# print housing
print '1'
# print housing_num
housing_prepared = full_pipeline.fit_transform(housing)
print housing_prepared
print housing_prepared.shape

# print(help(num_pipeline))
# print(help(full_pipeline))


lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
print some_data
some_labels = housing_labels.iloc[:5]
print some_labels
some_data_prepared = full_pipeline.transform(some_data)
print some_data_prepared
print("Predictions:\t" ,lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))



housing_predictions = lin_reg.predict(housing_prepared)
print housing_predictions
lin_mse = mean_squared_error(housing_labels, housing_predictions)
print lin_mse
lin_rmse = np.sqrt(lin_mse)
print lin_rmse




tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


housing_predictions = tree_reg.predict(housing_prepared)
print housing_predictions
tree_mse = mean_squared_error(housing_labels, housing_predictions)
print tree_mse
tree_rmse = np.sqrt(tree_mse)
print tree_rmse



scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
print rmse_scores



def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(rmse_scores)

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)