import numpy as np

# really need to figure out the syntax for einsum to avoid this
def col_by_row_mult(u,v):
    n = np.size(u)
    if np.size(v) == n and u.ndim == 1 and v.ndim == 1:
        A = np.zeros((n,n))
        for i in range(n):
            A[i,:] = u[i]*v
        return A
    raise Exception('Arguments must be one dimensional vectors of the same size!')

def get_low_rank_svd(U, V, Sigma, rank):
    m, n = U.shape[0], V.shape[1]
    Ak = np.zeros((m, n))
    for i in range(rank):
        Ak += Sigma[i]*col_by_row_mult(U[:,i], V[:,i])
    return Ak

def get_document_matrix(Sigma, V_T, rank):
    return np.diag(Sigma[:rank]) @ V_T[:rank]

def get_query(U, q, rank):
    return U[:,:rank].T @ q

def get_cosines(U, Sigma, V_T, q, rank):
    identity = np.eye(V_T.shape[1]) # nxn identity matrix
    Dk = get_document_matrix(Sigma, V_T, rank)
    qk = get_query(U, q, rank)
    cosines = np.zeros(Dk.shape[1])
    for j in range(len(cosines)):
        temp = Dk @ identity[:,j]
        cosines[j] = np.dot(qk, temp)/(np.linalg.norm(qk)*np.linalg.norm(temp))
    return cosines


if __name__ == '__main__':
    A = np.array([[0,0,0,1,0],
                [0,0,0,0,1],
                [0,0,0,0,1],
                [1,0,1,0,0],
                [1,0,0,0,0],
                [0,1,0,0,0],
                [1,0,1,1,0],
                [0,1,1,0,0],
                [0,0,1,1,1],
                [0,1,1,0,0]])
    q = np.array([0,0,0,0,0,0,0,1,1,1])
    U, Sigma, V_T = np.linalg.svd(A, full_matrices=False)
    '''print(f'shape of U: {U.shape}')
    print(f'shape of Sigma: {Sigma.shape}')
    print(f'shape of V.T: {V_T.shape}')'''
    '''SVD_test = U @ np.diag(Sigma) @ V_T
    SVD_test[np.abs(SVD_test) < 10e-12] = 0
    print(SVD_test)'''

    rank = A.shape[1]
    m, n = A.shape
    cosines = np.zeros((rank,n))
    for k in range(rank):
        cosines[k] = get_cosines(U, Sigma, V_T, q, k+1)
        print(f'cosine vector for subspace approximation A_{k+1}: \n{cosines[k]}\n')
