from DataManipulations import *
import pandas as pd
import os
from DataManipulationsOnTwoColumns import *

csv_path = os.path.join("datasets/Short_try", "Short_data.csv")
df = pd.read_csv(csv_path, low_memory=False)

df = removeNanColumns(df)
df = fixLowercase(df)
# dfInverse = df.stack()
# # print df
# # print dfInverse

encoded_df = encodeData(df, "true")
# print encoded_df

columnsICantUse = cleanData(encoded_df)
print columnsICantUse
for col in columnsICantUse:
    print col
    del encoded_df[col]


dfClean = removeRowMiss(encoded_df)
dfNorm = checkSimilarity(dfClean)
# dfNorm2 = dfNorm[:]
# dfWithoutDouble = printAllUnique(dfNorm, True) I have to fix this one looks like it is not operational
dfOneTime = checkColumnsMultiplyNames(dfNorm)
interestingSubjects = strongCorrOfData(dfNorm, 70)

whatToInquiry = raw_input("Please the number of the correlation you want to inquiry")
print "Working on %s and %s"%(interestingSubjects[whatToInquiry][0], interestingSubjects[whatToInquiry][1])

dfTwoColumns = pd.concat([df[interestingSubjects[whatToInquiry][0]], df[interestingSubjects[whatToInquiry][1]]], axis=1)
dfTwoColumns, dictOfEncoding = twoEncoder(dfTwoColumns, "true")

chosenCol = checkSmallestDistribution(dfTwoColumns)
twoColDetails(dfTwoColumns, dictOfEncoding, chosenCol)


