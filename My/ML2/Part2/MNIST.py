from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier

mnist = fetch_mldata('MNIST original')
print mnist


X, y = mnist["data"], mnist["target"]
print X.shape
print y.shape


some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
plt.axis("off")
plt.show()
print y[36000]
print X[36000]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

print sgd_clf
print sgd_clf.predict([some_digit])

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    print y_pred
    n_correct = sum(y_pred == y_test_fold)
    print float(n_correct)/len(y_pred)

from sklearn.model_selection import cross_val_score
print cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
print cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')

from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix
print confusion_matrix(y_train_5, y_train_pred)
confusionMatrix = confusion_matrix(y_train_5, y_train_pred)

####Precision and Recall
from sklearn.metrics import precision_score, recall_score
truePositive = float(confusionMatrix[1, 1])
falseNegative = float(confusionMatrix[1, 0])
falsePositive = float(confusionMatrix[0, 1])
print truePositive, falseNegative, falsePositive


precisionScore = truePositive/(truePositive + falsePositive)
recallScore = truePositive/(truePositive + falseNegative)
f1Score = 2*((precisionScore*recallScore)/(precisionScore+recallScore))
print recallScore, precisionScore, f1Score
# precision_score(y_train_5, y_pred)
# print recall_score(y_train_5, y_train_pred)

####Precision/Recall Tradeoff
y_scores = sgd_clf.decision_function([some_digit])
print y_scores
threshold = 0
y_some_digit_pred = (y_scores > threshold)
print y_some_digit_pred

threshold = 200000
y_some_digit_pred = (y_scores > threshold)
print y_some_digit_pred

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
# print y_scores

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
print precisions, recalls, thresholds

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

y_train_pred_90 = (y_scores > 70000)
print precision_score(y_train_5, y_train_pred_90)
print recall_score(y_train_5, y_train_pred_90)

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
print fpr, tpr, thresholds

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plot_roc_curve(fpr, tpr)
plt.show()

from sklearn.metrics import roc_auc_score
print roc_auc_score(y_train_5, y_scores)

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")
print y_probas_forest

y_scores_forest = y_probas_forest[:, 1]
print y_scores_forest
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr,tpr, "b:", label="SDG")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="bottom right")
plt.show()
print roc_auc_score(y_train_5, y_scores_forest)