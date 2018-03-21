from FetchSQL import *
import pandas as pd
import os

csv_path = os.path.join("datasets/Short_try", "Short_data.csv")
df = pd.read_csv(csv_path, low_memory=False)

df = removeNanColumns(df)
df = fixLowercase(df)
encoded_df = encodeData(df, "true")
print encoded_df

# columnsICantUse = cleanData(df)
# print columnsICantUse
# for col in columnsICantUse:
#     print col
#     del df[col]


# # dfClean = removeRowMiss(encoded_pd)
# # dfNorm = checkSimilarity(dfClean)
# # dfNorm2 = dfNorm[:]
# # # dfWithoutDouble = printAllUnique(dfNorm, True)
# # strongCorrOfData(dfNorm, 90)
# # # print dfWithoutDouble