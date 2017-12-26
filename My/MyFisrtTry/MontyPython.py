import sqlite3
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from FetchSQL import cleanData, encodeData, removeRowMiss, checkSimilarity, printAllUnique, strongCorrOfData
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from FetchSQL import DataFrameSelector
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_transmission_per_record = True):
        self.add_transmission_per_record = add_transmission_per_record
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        series_per_record_data = X["series"]/X["record_date"]
        if self.add_transmission_per_record:
            return np.c_[X, series_per_record_data]
        else:
            return np.c_[X]


conn = sqlite3.connect("datasets/Monty_Python_Flying_Circus/database.sqlite")
res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print name[0]
df = pd.read_sql(con=conn, sql='select * from scripts')
conn.close()

columnsICantUse = cleanData(df)
for col in columnsICantUse:
    del df[col]


encoded_pd = encodeData(df)
dfClean = removeRowMiss(encoded_pd)
dfNorm = checkSimilarity(dfClean)
dfNorm2 = dfNorm[:]
dfWithoutDouble = printAllUnique(dfNorm, True)
strongCorrOfData(dfWithoutDouble, 90)


# train_set, test_set = train_test_split(dfWithoutDouble, test_size=0.2)
# print train_set
# dfWithoutDouble.plot(kind="scatter", x="transmission_date", y="episode_name", alpha=0.1, s=dfWithoutDouble["record_date"]*100)


# attr_adder = CombinedAttributesAdder(add_transmission_per_record=True)
# dfExtra = attr_adder.transform(dfWithoutDouble)
# print dfExtra
#
num_pipeline = Pipeline([
    # ('selector', DataFrameSelector(list(dfWithoutDouble))),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
dfAfterPipeline = num_pipeline.fit_transform(dfWithoutDouble)
# print dfAfterPipeline

comData = dfNorm2["episode"]
lin_reg = LinearRegression()
lin_reg.fit(dfAfterPipeline, comData)


some_data = dfWithoutDouble.iloc[:5]
some_labels = comData.iloc[:5]
some_data_prepared = num_pipeline.transform(some_data)
print ("Predictions:\t", lin_reg.predict(some_data_prepared))
print ("Labels:\t\t", list(some_labels))

show_prediction = lin_reg.predict(dfAfterPipeline)
lin_mse = mean_squared_error(comData, show_prediction)
lin_rmse = np.sqrt(lin_mse)
print lin_rmse

# dfWithoutDouble.hist(bins=50)
# plt.show()


