import numpy as np
from copy import deepcopy

def normalize_data(X, d):
    X_norm = deepcopy(X)
    for col in range(d):
        X_norm[:,col] -= np.mean(X_norm[:,col])
        X_norm[:,col] /= np.std(X_norm[:,col])
    return X_norm

def normalize_data2(X, d):
    X_norm = deepcopy(X)
    for col in range(d):
        X_norm[:,col] /= np.linalg.norm(X_norm[:,col])
    return X_norm

def fill_nan(X):
    for i in range(X.shape[1]):
        col_mean = np.mean(X[:,i])
        np.nan_to_num(X[:,i], copy=False, nan=col_mean)

# TODO: add method to fill nan values
# TODO: feature selection (PCA), possibly
# TODO: get rank of data matrix and check singularity
    # implement more than one method to solve for the weight matrix