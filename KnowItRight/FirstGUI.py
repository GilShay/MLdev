from easygui import *
import csv,os
import numpy as np



def getFile():
    fileLocation = fileopenbox("Please choose the file you want us to enquire", "Choose a file")
    print fileLocation
    if not fileLocation.endswith("csv"):
        msgbox("Sorry we supporting only csv files at the moment")
        exit("1")
    msgbox("First check pass, starting second test")
    return fileLocation


def checkFileContentsCSV(fileLocation):
    firstDataStorage = []
    with open(os.path.join(fileLocation), "rb") as myFile:
        myFileReader = csv.reader(myFile)
        for row in myFileReader:
            firstDataStorage.append(row)
        strToCheck = firstDataStorage[0]
        strToCheck = str(strToCheck)
        if strToCheck == r"['\xef\xbb\xbf']" or not firstDataStorage[0]:
            msgbox("Your file is empty")
            exit("1")
        firstDataStorageTrans = np.transpose(firstDataStorage)
        print firstDataStorage
        print firstDataStorageTrans
        if len(firstDataStorageTrans)<6:
            choices = ["Yes", "No"]
            reply = choicebox("It seems like you data has less then six events, do you want to continue?", choices=choices)
        if reply == "No":
            msgbox("OK exiting")
            exit(1)
        elif reply == "Yes":
            msgbox("OK we are on it")


if __name__ == "__main__":
    msgbox("Welcome to KIR", "KIR", "Start")
    fileLocation = getFile()
    checkFileContentsCSV(fileLocation)
