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
print grid.cv_results_.keys()
print grid.cv_results_['std_test_score']
grid_mean_scores = grid.cv_results_['mean_test_score']
print grid_mean_scores
print grid.cv_results_['params']

plt.plot(k_range, grid_mean_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

print grid.best_score_
print grid.best_params_
print grid.best_estimator_
print grid.best_index_


#searching multiple parameters simultaneously
weight_options = ['uniform', 'distance']

param_grid = dict(n_neighbors=k_range, weights=weight_options)
print param_grid

grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
print grid
print grid.fit(X,y)
print grid.cv_results_['std_test_score']
grid_mean_scores = grid.cv_results_['mean_test_score']
print grid_mean_scores
print grid.cv_results_['params']
print grid.best_score_
print grid.best_params_
print grid.best_estimator_
print grid.best_index_

knn = KNeighborsClassifier(n_neighbors=13, weights='uniform')
print knn
print knn.fit(X,y)
print knn.predict([[3, 5, 4, 2]])
print grid.predict([[3, 5, 4, 2]])

from sklearn.model_selection import RandomizedSearchCV

param_dist = dict(n_neighbors=k_range, weights=weight_options)

rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
print rand.fit(X,y)
print rand.cv_results_['std_test_score']
grid_mean_scores = rand.cv_results_['mean_test_score']
print grid_mean_scores
print rand.cv_results_['params']
print rand.best_score_
print rand.best_params_
print rand.best_estimator_
print rand.best_index_

best_score = []
for _ in range(20):
     rand = RandomizedSearchCV(knn, param_dist, cv=10, scoring='accuracy', n_iter=10)
     rand.fit(X, y)
     best_score.append(round(rand.best_score_, 3))
print best_score