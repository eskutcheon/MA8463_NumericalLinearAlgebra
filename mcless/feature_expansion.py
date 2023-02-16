import numpy as np
from mcless import mcless
from scipy.optimize import minimize

# Taylor expansion of Gaussian kernel
def get_Taylor(x, sigma):
    num_var = len(x)
    phi_x = np.zeros(num_var)
    mult = np.exp(-((x/sigma)**2)/2)
    for i in range(num_var):
        phi_x[i] = np.dot((1/np.math.factorial(i))*((x/sigma)**i), mult)
    return phi_x

def multi_Taylor_exp(X, N, d):
    X_new = np.zeros((N,d))
    new_features = np.zeros((N,d))
    for i in range(N):
        X_new[i,:] = get_Taylor(X[i,:], 0.25)
        # covers d different points p of the form [p_j, p_j, ..., p_j] for ||x-p||
        for j in range(d):
            new_features[i,j] = np.linalg.norm(X[i,:] - X_new[i,j])
    return np.concatenate((X, new_features), axis=1)

def single_Taylor_exp(X, N, d):
    X_new = np.zeros((N,d))
    new_features = np.zeros((N,1))
    # covers a single point p for each sample so that the new feature is ||x-p||
    for i in range(N):
        X_new[i,:] = get_Taylor(X[i,:], 0.25)
        new_features[i,:] = np.linalg.norm(X[i,:] - X_new[i,:])
    return np.concatenate((X, new_features), axis=1)

def single_col_mean_exp(X, N, d):
    new_feature = np.zeros((N,1))
    column_mean = np.array([np.mean(X[:,i]) for i in range(d)])
    for i in range(N):
        new_feature[i] = np.linalg.norm(X[i,:] - column_mean)
    return np.concatenate((X, new_feature), axis=1)

def single_col_med_exp(X, N, d):
    new_feature = np.zeros((N,1))
    column_mean = np.array([np.median(X[:,i]) for i in range(d)])
    for i in range(N):
        new_feature[i] = np.linalg.norm(X[i,:] - column_mean)
    return np.concatenate((X, new_feature), axis=1)

# lots of issues with this logic, but time doesn't permit testing of a new loss function
def mse_loss_exp(A, y, W, N, d):
    # y: scalar, a: vector of size (d+1), w: vector of size (d+1)
    '''print(f'shape of A: {A.shape}')
    print(f'shape of W_hat: {W.shape}')
    print(f'shape of y: {y.shape}')
    print(f'N,d = {N,d}')'''
    grad = lambda a, w, y, a_i: (-2*a_i)*(y - np.dot(a,w))
    new_feature = np.zeros((N,len(W[0,:])))
    for k in range(len(W[0,:])):
        for i in range(N):
            p = np.array([grad(A[i,:], W[:,k], y[i], A[i,j]) for j in range(d)])
            new_feature[i,k] = np.linalg.norm(A[i,:] - p)
    return new_feature

def test_mse_features(model, X_train, X_test, y_train, y_test):
    N, d = X_test.shape
    X_new = np.concatenate((np.ones((N,1)), X_test), axis=1)
    new_train_features = mse_loss_exp(model.A, y_train, model.W_hat, *model.A.shape)
    # I realize the logical issue with the line below - I'm leaving this method in either way
        # to show that I understand p should probably be the maximum decrease in a loss function
    new_test_features = mse_loss_exp(X_new, y_test, model.W_hat, N, d+1)
    X_train_new = np.concatenate((model.A, new_train_features), axis=1)
    X_test_new = np.concatenate((X_new, new_test_features), axis=1)
    fe_model = mcless(X_train_new, y_train)
    fe_model.compute_training_matrices()
    return fe_model.predict(X_test_new)

def test_taylor_features(X_train, y_train, X_test):
    X_train_single = single_Taylor_exp(X_train, *X_train.shape)
    X_train_multi = multi_Taylor_exp(X_train, *X_train.shape)
    X_test_single = single_Taylor_exp(X_test, *X_test.shape)
    X_test_multi = multi_Taylor_exp(X_test, *X_test.shape)
    single_model = mcless(X_train_single, y_train)
    single_model.compute_training_matrices()
    multi_model = mcless(X_train_multi, y_train)
    multi_model.compute_training_matrices()
    return single_model.predict(X_test_single), multi_model.predict(X_test_multi)

def test_mean_features(X_train, y_train, X_test):
    X_train_new = single_col_mean_exp(X_train, *X_train.shape)
    X_test_new = single_col_mean_exp(X_test, *X_test.shape)
    model = mcless(X_train_new, y_train)
    model.compute_training_matrices()
    return model.predict(X_test_new)

def test_med_features(X_train, y_train, X_test):
    X_train_new = single_col_med_exp(X_train, *X_train.shape)
    X_test_new = single_col_med_exp(X_test, *X_test.shape)
    model = mcless(X_train_new, y_train)
    model.compute_training_matrices()
    return model.predict(X_test_new)