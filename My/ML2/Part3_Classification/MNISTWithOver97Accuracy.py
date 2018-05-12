from sklearn.datasets import fetch_mldata
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

mnist = fetch_mldata('MNIST original')
X ,y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# y_train_5 = (y_train == 5)
# y_test_5 = (y_test == 5)
# knn_clf = KNeighborsClassifier(n_neighbors=3)
param_grid = [{'weights': ["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_train_5)
# y_pred = knn_clf.predict(X_test)
# n_correct = sum(y_pred == y_test_5)
# print float(n_correct)/len(y_pred)
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
print grid_search.fit(X_train, y_train)
print grid_search.best_params_
print grid_search.best_score_
from sklearn.metrics import accuracy_score
y_pred = grid_search.predict(X_test)
print accuracy_score(y_test, y_pred)
