from FetchSQL import cleanData,encodeData, removeRowMiss, checkSimilarity, printAllUnique, strongCorrOfData
import pandas as pd
import os

csv_path = os.path.join("datasets/Global_Terrorism", "globalterrorismdb_0617dist.csv")
df = pd.read_csv(csv_path, low_memory=False)


columnsICantUse = cleanData(df)
for col in columnsICantUse:
    print col
    del df[col]


encoded_pd = encodeData(df)
# print encoded_pd
dfClean = removeRowMiss(encoded_pd)
dfNorm = checkSimilarity(dfClean)
dfNorm2 = dfNorm[:]
# dfWithoutDouble = printAllUnique(dfNorm, True)
strongCorrOfData(dfNorm, 90)
# print dfWithoutDouble