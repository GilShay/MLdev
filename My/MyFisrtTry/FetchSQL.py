import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encodeData(pdData):
    for col in pdData:
        if type(pdData[col][1]) == str:
            encoder = LabelEncoder()
            colIscat = pdData[col]
            colIscat = encoder.fit_transform(colIscat)
            # pdData = pdData.drop(col, 1)
            # pdData.insert(loc=0, column=col, value=colIscat)
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
print encoded_pd
