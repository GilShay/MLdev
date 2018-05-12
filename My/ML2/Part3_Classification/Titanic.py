import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import math
import numpy as np

dfTiranicTrain = pd.read_csv(r"C:\Users\hilla\Documents\MLdev\My\ML2\Part2\dataset\Titanic\train.csv", index_col=0)
dfTiranicTest = pd.read_csv(r"C:\Users\hilla\Documents\MLdev\My\ML2\Part2\dataset\Titanic\test.csv", index_col=0)
dfTiranicAnswers = pd.read_csv(r"C:\Users\hilla\Documents\MLdev\My\ML2\Part2\dataset\Titanic\answers.csv", index_col=0)

dfTiranicTrain = dfTiranicTrain.drop(["Name", "Ticket", "Cabin"], axis=1)
# print dfTiranicTrain.info()
# print dfTiranicTest.info()
X_test = dfTiranicTest[['Pclass', 'SibSp', 'Parch']]
clf_knn = KNeighborsClassifier(n_neighbors=30)



# X = dfTiranicTrain[['Pclass', 'Sex', 'SibSp', 'Parch', 'Age', 'Fare', 'Embarked']]
X = dfTiranicTrain[['Pclass', 'SibSp', 'Parch']]
y = dfTiranicTrain['Survived']
# print X.head
# print y.head
clf_knn.fit(X, y)
print X_test.info()
print dfTiranicAnswers.info()
y_pred = clf_knn.predict(X_test)
print metrics.accuracy_score(dfTiranicAnswers, y_pred)