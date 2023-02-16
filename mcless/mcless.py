import numpy as np

class mcless(object):
    def __init__(self, X, y):
        self.X = X
        self.labels = y
        self.N, self.d = X.shape
        self.num_classes = len(set(y))

    def get_information_matrix(self):
        self.A = np.concatenate((np.ones((self.N,1)), self.X), axis=1)
        # print(f'shape of A: {self.A.shape}')

    '''# initialize weights
    def get_weight_matrix(self):
        self.W = np.random.rand(self.d+1, self.num_classes)
        # print(f'shape of W: {self.W.shape}')'''

    def get_source_matrix(self):
        self.B = np.zeros((self.N, self.num_classes))
        for i in range(self.N):
            self.B[i,self.labels[i]] = 1
        # print(f'shape of B: {self.B.shape}')

    def get_SVD(self):
        U, Sigma, V_T = np.linalg.svd(self.A, full_matrices=False)
        self.SVD = (U, np.diag(Sigma), V_T)

    def get_pseudoinverse(self):
        self.A_plus = self.SVD[2].T @ np.diag((1/np.diag(self.SVD[1]))) @ self.SVD[0].T
        # print(f'shape of A_plus: {self.A_plus.shape}')

    def get_min_norm(self):
        self.W_hat = self.A_plus @ self.B
        # print(f'shape of W_hat: {self.W_hat.shape}')

    def compute_training_matrices(self):
        self.get_information_matrix()
        self.get_source_matrix()
        self.get_SVD()
        self.get_pseudoinverse()
        self.get_min_norm()

    def predict(self, X_test):
        m, n = X_test.shape
        A_test = np.concatenate((np.ones((m,1)), X_test), axis=1)
        return A_test @ self.W_hat

    # TODO: check rank of A and possibly implement rank-deficient solution
    # TODO: refer to page 153-154 of textbook for solving LS problems by partitioning
    # TODO: implement feature selection within mcless - check PCA on pg 156