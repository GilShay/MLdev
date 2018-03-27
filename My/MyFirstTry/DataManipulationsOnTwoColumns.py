import pandas as pd
from sklearn.preprocessing import LabelEncoder

def printWhatIHave(df):
    print df

def twoEncoder(pdData, flag):
    dictOfEncoding = {}
    for col in pdData:
        if flag:
            if pdData[col].dtype == "object":
                encoder = LabelEncoder()
                colIscat = pdData[col]
                encoder = encoder.fit(colIscat)
                colIscat = encoder.transform(colIscat)
                pdData = pdData.drop(col, 1)
                pdData.insert(loc=0, column=col, value=colIscat)
                dictOfEncoding[pdData[col].name] = list(encoder.classes_)
        else:
            encoder = LabelEncoder()
            colIscat = pdData[col]
            encoder = encoder.fit(colIscat)
            colIscat = encoder.transform(colIscat)
            pdData = pdData.drop(col, 1)
            pdData.insert(loc=0, column=col, value=colIscat)
            dictOfEncoding[pdData[col].name] = list(encoder.classes_)
    return pdData, dictOfEncoding


def twoColDetails(dfTwoCol, dictOfEncoding, chosenCol):
    for col in dfTwoCol:
        if col is not chosenCol:
            notChosenCol = col
    detailDict = {}
    counterList = []
    for value in list(dfTwoCol[chosenCol].unique()):
        if chosenCol in dictOfEncoding:
            valueName = dictOfEncoding[chosenCol][value]
        else:
            valueName = value
        for counter in range(0, len(dfTwoCol.index)):
            if dfTwoCol[chosenCol][counter] == value:
                if notChosenCol in dictOfEncoding:
                    valueNameNot = dictOfEncoding[notChosenCol][counter]
                else:
                    valueNameNot = dfTwoCol[notChosenCol][counter]
                counterList.append(valueNameNot)
        detailDict[valueName] = counterList
        counterList = []
    print detailDict
