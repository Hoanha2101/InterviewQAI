import numpy as np
from utils import *

def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two points.

    Parameters:
    - a: 1D array, first point.
    - b: 1D array, second point.

    Returns:
    - float, distance between a and b.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def predict(X_train, y_train, X_test, k=3):
    """
    Predict labels for test samples using k-NN.

    Parameters:
    - X_train: 2D array, training samples.
    - y_train: 1D array, training labels.
    - X_test: 2D array, test samples.
    - k: int, number of nearest neighbors (default is 3).

    Returns:
    - 1D array, predicted labels for test samples.
    """
    
    y_pred = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        distances = np.array([euclidean_distance(X_test[i], x) for x in X_train])
        k_indices = np.argsort(distances)[:k]
        k_nearest_labels = y_train[k_indices]
        y_pred[i] = np.argmax(np.bincount(k_nearest_labels))
        
    return y_pred

