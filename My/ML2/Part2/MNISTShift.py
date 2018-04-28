from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage.interpolation import shift
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

mnist = fetch_mldata('MNIST original')
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

def plot_digit(image):
    some_digit_image = image.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation= 'nearest' )
    plt.axis('off')
    plt.show()

def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

some_number = raw_input("Please insret number between 0 and 70000")
some_number = int(some_number)
image = X[some_number]
plot_digit(image)
image = shift_image(image, 5, 5)
plot_digit(image)

X_train_augmented = [image for image in X_train]
y_train_augmented = [image for image in y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier()
print knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)
print accuracy_score(y_test, y_pred)
