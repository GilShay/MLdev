from DataManipulations import *
import pandas as pd
import os

csv_path = os.path.join("datasets/Short_try", "Short_data.csv")
df = pd.read_csv(csv_path, low_memory=False)

df = removeNanColumns(df)
df = fixLowercase(df)
encoded_df = encodeData(df, "true")

columnsICantUse = cleanData(encoded_df)
print columnsICantUse
for col in columnsICantUse:
    print col
    del encoded_df[col]


dfClean = removeRowMiss(encoded_df)
print dfClean
dfNorm = checkSimilarity(dfClean)
print dfNorm
# dfNorm2 = dfNorm[:]
# dfWithoutDouble = printAllUnique(dfNorm, True) I have to fix this one looks like it is not operational
dfOneTime = checkColumnsMultiplyNames(dfNorm)
print dfOneTime
strongCorrOfData(dfNorm, 70)
# print dfWithoutDouble