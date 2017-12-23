import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer, normalize, Normalizer
import matplotlib.pyplot as plt


def encodeData(pdData):
    for col in pdData:
        pdData = pdData[pd.notnull(pdData[col])]
        print pdData[col][1]
        # if pdData[col][1].isdigit():
        encoder = LabelEncoder()
        colIscat = pdData[col]
        colIscat = encoder.fit_transform(colIscat)
        pdData = pdData.drop(col, 1)
        pdData.insert(loc=0, column=col, value=colIscat)
        # print colIscat
    return pdData

def cleanData(df):
    allDataCount = df.count().min()
    columnsICantUse = []
    for paramNames in list(df):
        for count in df[paramNames].value_counts():
            if count < allDataCount * 0.01:
                columnsICantUse.append(paramNames)
                break
    return columnsICantUse

def removeRowMiss(df):
    imputer = Imputer(strategy='median')
    imputer.fit(df)
    print imputer.statistics_
    X = imputer.transform(df)
    df = pd.DataFrame(X, columns=df.columns)
    # for col in df:
    #     df.dropna(subset=[col])
    return df

def printAllUnique(df):
    df = df.reset_index()
    for col in df:
        colUnique = []
        for i in range(0, len(df.index)):
            if df[col][i] not in colUnique:
                colUnique.append(df[col][i])
    print sorted(colUnique)




conn = sqlite3.connect("datasets/Monty_Python_Flying_Circus/database.sqlite")

res = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print name[0]

df = pd.read_sql(con=conn, sql='select * from scripts')
conn.close()

columnsICantUse = cleanData(df)
print columnsICantUse
for col in columnsICantUse:
    del df[col]
print df
encoded_pd = encodeData(df)
dfClean = removeRowMiss(encoded_pd)
print dfClean

temp1 = dfClean["transmission_date"]
temp2 = dfClean["episode"]

df_norm = (temp1 - temp1.mean()) / (temp1.max() - temp1.min())
printAllUnique(df_norm)
df_norm = (temp2 - temp2.mean()) / (temp2.max() - temp2.min())
printAllUnique(df_norm)

# printAllUnique(dfClean)
# temp = dfClean["record_date"]
# normData = normalize(temp, axis=0)
# print normData
# print dfClean["episode_name"]
# normData = pd.DataFrame(normData)
# printAllUnique(normData)

# print type(normData)
# print type(dfClean)
#
# dfClean.hist(bins=50)
# plt.show()


