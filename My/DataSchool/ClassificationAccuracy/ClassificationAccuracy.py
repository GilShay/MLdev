import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os

# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
currentLocation= os.getcwd()
dataLocation = os.path.join(currentLocation, "datasets\diabetes.csv")
# col_names = ['pregnat', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(dataLocation, header=0)
print pima.head()

feature_cols = ['Pregnancies', 'Insulin', 'BMI', 'Age']
X = pima[feature_cols]
y = pima.Outcome

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_class = logreg.predict(X_test)
print metrics.accuracy_score(y_test, y_pred_class)

print y_test.value_counts()
print y_test.mean()
print 1 - y_test.mean()
print max(y_test.mean(), 1 - y_test.mean())
print y_test.value_counts().head(1)/ len(y_test)


print "True:", y_test.values[0:25]
print "Pred:", y_pred_class[0:25]

print metrics.confusion_matrix(y_test, y_pred_class)

confusion = metrics.confusion_matrix(y_test, y_pred_class)
TP = confusion[1,1]
TN = confusion[0,0]
FP = confusion[0,1]
FN = confusion[1,0]

print (TP + TN)/ float(TP + TN + FP + FN)
print metrics.accuracy_score(y_test, y_pred_class)

print (FP + FN)/ float(TP + TN + FP + FN)
print 1- metrics.accuracy_score(y_test, y_pred_class)

print TP / float(TP + FN)
print metrics.recall_score(y_test,y_pred_class)

print TN / float(TN + FP)

print FP / float(TN + FP)

print TP / float(TP + FP)
print metrics.precision_score(y_test,y_pred_class)


