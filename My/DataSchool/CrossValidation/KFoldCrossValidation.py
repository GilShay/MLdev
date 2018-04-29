from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

iris = load_iris()
X = iris.data
y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y)
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print scores
print scores.mean()
k_range = range(1,31)
k_scores = []
for k in k_range:
     knn = KNeighborsClassifier(n_neighbors=k)
     scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
     k_scores.append(scores.mean())
print k_scores
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross valudation accuracy')
plt.show()

knn = KNeighborsClassifier(n_neighbors=20)
print cross_val_score(knn, X, y, cv=10, scoring="accuracy").mean()

logreg = LogisticRegression()
print cross_val_score(logreg, X, y, cv=10, scoring="accuracy").mean()


####GridSearchCV
from sklearn.model_selection import GridSearchCV

k_range = range(1,31)
print k_range

param_grid = dict(n_neighbors=k_range)
print param_grid
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
print grid.fit(X,y)
print grid.cv_results_

print grid.cv_results_.keys()