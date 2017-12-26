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


def encodeData(pdData):
    for col in pdData:
        # pdData = pdData[pd.notnull(pdData[col])]
        # print pdData[col][1]
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


def strongCorrOfData(df, percent):
    percent = percent*0.01
    corr_matrix = df.corr()
    for col in corr_matrix:
        for i in range(0, len(corr_matrix[col])):
            if (abs(corr_matrix[col][i]) > percent) & (corr_matrix[col][i] != 1):
                print ("There is corrolation between %s and %s about %s"%(col, corr_matrix.index[i], corr_matrix[col][i]))

