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

print logreg.predict(X_test)[0:10]
print logreg.predict_proba(X_test)[0:10, :]
print logreg.predict_proba(X_test)[0:10, 1]

y_pred_prob = logreg.predict_proba(X_test)[:, 1]

import matplotlib.pyplot as plt
# plt.rcParams['font.size'] = 14
# plt.hist(y_pred_prob)
# plt.xlim(0, 1)
# plt.title('Histogram of predicted probabilities')
# plt.xlabel('Predicted probability of diabetes')
# plt.ylabel('Frequency')
# plt.show()

y_pred_prob = logreg.predict_proba(X_test)
from sklearn.preprocessing import binarize
y_pred_class = binarize(y_pred_prob, 0.3)[:, 1]
print y_pred_prob[0:10, 1]
print y_pred_class[0:10]

print confusion
print metrics.confusion_matrix(y_test, y_pred_class)
print 46 / float(46 + 16)
print 80 / float(80 + 50)

y_pred_prob = logreg.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate(1 - Specificity)')
plt.ylabel('True Positive Rate(Sensitivity)')
plt.grid(True)
plt.show()

def evaluate_threshold(threshold):
    print 'Sensitivity:', tpr[thresholds > threshold][-1]
    print 'Specificity:', 1 - fpr[thresholds > threshold][-1]

evaluate_threshold(0.5)
evaluate_threshold(0.3)

print metrics.roc_auc_score(y_test, y_pred_prob)
from sklearn.model_selection import cross_val_score
print cross_val_score(logreg, X, y, cv=10, scoring='roc_auc').mean()