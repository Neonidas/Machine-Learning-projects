import numpy as np


def knn(X_train, y_train, X_test, k, dist):
    # The function will return the class for x based on its neighbours from the X_train
    # sample.
    def classify_single(x):
        # Here we create an array of distances from x to each of the X_train objects.
        dists = [dist(x, xprime) for xprime in X_train]
        # This array will contain the indices of k nearest to the x objects. NumPy.argpartition
        # might be useful here.
        indices = np.argpartition(dists, k)
        # The function returns the most frequent class among those in y_train represented
        # by the indices.
        k_nearest = y_train[indices][:k]
        return np.argmax(np.bincount(k_nearest))

    return [classify_single(x) for x in X_test]
