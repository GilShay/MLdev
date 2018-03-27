import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameSelector(BaseEstimator, TransformerMixin):#Pandas DataFrame to numpy
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

def removeNanColumns(df):
    for i in df:
        iCount = 0
        for value in df[i]:
            if value == ("Nan" or "nan"  or "Null"):
                iCount = iCount + 1
        X = df[i].isna()
        for na in X:
            if na:
                iCount = iCount + 1
        if iCount > 0.5*df[i].size:
            del df[i]
            print "Removing %s column missing at list half of data"%i
    return df

def fixLowercase(df):
    for i in df:
        if df[i].dtype == "object":
            df[i] = df[i].str.lower()
    return df

def encodeData(pdData, flag):
    for col in pdData:
        if flag:
            if pdData[col].dtype == "object":
                encoder = LabelEncoder()
                colIscat = pdData[col]
                colIscat = encoder.fit_transform(colIscat)
                pdData = pdData.drop(col, 1)
                pdData.insert(loc=0, column=col, value=colIscat)
        else:
            encoder = LabelEncoder()
            colIscat = pdData[col]
            colIscat = encoder.fit_transform(colIscat)
            pdData = pdData.drop(col, 1)
            pdData.insert(loc=0, column=col, value=colIscat)
    return pdData


def cleanData(df):
    allDataCount = df.count().min()
    columnsICantUse = []
    for paramNames in list(df):
        for count in df[paramNames].value_counts():
            if count < allDataCount * 0.01 and count == 1:
                columnsICantUse.append(paramNames)
                break
    return columnsICantUse


def removeRowMiss(df):
    imputer = Imputer(strategy='median')
    imputer.fit(df)
    # print imputer.statistics_
    # for col in df:
    #     if len(df[col]) == 1:
    #         del df[col]
    X = imputer.transform(df)
    df = pd.DataFrame(X, columns=df.columns)
    # for col in df:
    #     df.dropna(subset=[col])
    return df


def printAllUnique(df, flag):
    dfData = df
    allUnique = []
    for col in df:
        colUnique = []
        for i in range(0, len(df.index)):
            if df[col][i] not in colUnique:
                colUnique.append(df[col][i])
        if flag:
            for i in range(0, len(allUnique)):
                   if allUnique[i] == colUnique:
                       del dfData[col]
                       print ("We droped %s"%col)
        print colUnique
        allUnique.append((sorted(colUnique[:])))
    return dfData


def checkSimilarity(df):
    df_norm = pd.DataFrame()
    for col in df:
        oneCol = (df[col] - df[col].mean()) / (df[col].max() - df[col].min())
        df_norm = df_norm.append(oneCol)
    df_norm = df_norm.T
    return df_norm


# def checkSimilarity2(df):
#     df_norm = pd.DataFrame()
#     scaler = StandardScaler()
#     for col in df:
#         oneCol = scaler.fit(df[col].T)
#         df_norm = df_norm.append(oneCol)
#     df_norm = df_norm.T
#     return df_norm


def checkColumnsMultiplyNames(df):
    colNames = list(df)
    dfOneTime = df
    for i in df:
        temp = i + ".1"
        if temp in colNames:
            print "I can't work like this you gave me several columns with the same name, so removing %s once"%i
            del dfOneTime[temp]
    return dfOneTime


def strongCorrOfData(df, percent):
    percent = percent*0.01
    corr_matrix = df.corr()
    myCounter = 0
    anotherCounter = 0
    interestingSubjects = {}
    for col in corr_matrix:
        for i in range(myCounter, len(corr_matrix[col])):
            if (corr_matrix[col][i] == 1):
                if col is not corr_matrix.index[i]:
                    print ("%s and %s have the same values, I'm not taking them into account"%(col, corr_matrix.index[i]))
            if (abs(corr_matrix[col][i]) > percent) & (corr_matrix[col][i] != 1):
                anotherCounter = anotherCounter + 1
                strCounter = str(anotherCounter)
                print ("%s. There is corrolation between %s and %s about %s"%(anotherCounter, col, corr_matrix.index[i], corr_matrix[col][i]))
                interestingSubjects[strCounter] = [col, corr_matrix.index[i]]
        myCounter = myCounter + 1
    return interestingSubjects


def checkSmallestDistribution(df):
    distributionCounter = 0
    masterCounter = float("inf")
    dataCon = []
    for col in df:
        for i in df[col]:
            if i not in dataCon:
                distributionCounter = distributionCounter + 1
                dataCon.append(i)
        if distributionCounter < masterCounter:
            masterCounter = distributionCounter
            chosenCol = col
    return chosenCol

