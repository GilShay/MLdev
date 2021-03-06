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
# print y_train_pred
# print y_train_5

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
# print precision_score(y_train_5, y_pred)
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

####Multiclass Classification

sgd_clf.fit(X_train, y_train)
print sgd_clf.predict([some_digit])
some_digit_scores = sgd_clf.decision_function([some_digit])
print some_digit_scores
print np.argmax(some_digit_scores)
print sgd_clf.classes_
# print sgd_clf.classes[5]

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
print ovo_clf.predict([some_digit])
print len(ovo_clf.estimators_)

forest_clf.fit(X_train, y_train)
print forest_clf.predict([some_digit])
print forest_clf.predict_proba([some_digit])
print cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")



###Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
print row_sums
conf_mx = np.array(conf_mx, dtype=float)
print conf_mx
norm_conf_mx = conf_mx/ row_sums
print norm_conf_mx

np.fill_diagonal(norm_conf_mx, 0)
print norm_conf_mx
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

cl_a, cl_b =3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()


###Multilabrl Classification
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
print y_train_large
y_train_odd = (y_train % 2 == 1)
print y_train_odd
y_multilabel = np.c_[y_train_large, y_train_odd]
print y_multilabel
print X_train

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print knn_clf.predict([some_digit])


# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
# print f1Score(y_train, y_train_knn_pred, average="macro")

###Multiouput Classification
from numpy.random import randint
noise_train = randint(0, 100, (len(X_train), 784))
print noise_train
noise_test = randint(0, 100, (len(X_test), 784))
print noise_test
X_train_mod = X_train + noise_train
X_test_mod = X_test + noise_test
y_train_mod = X_train
y_test_mod = X_test

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[3600]])
# plot_digit(clean_digit)
# plot_digit(X_test[3600])
plt.figure(figsize=(2,2))
# plt.subplot(211); plot_digits(clean_digit, images_per_row=1)
old_digit = X_test[3600].reshape(28, 28)
plot_digit(clean_digit)
plt.show()
plot_digit(X_test[3600])
# plt.subplot(221); plot_digits(old_digit, images_per_row=1)
plt.show()
