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

def get_document_matrix(Sigma, V, rank):
    D = np.zeros((V.shape[0], rank))
    for i in range(rank):
        D[:,i] = Sigma[i]*V[:,i]
    return D

def get_query(U, q, rank):
    return U[:,:rank].T @ q

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
    SVD_test = U @ np.diag(Sigma) @ V_T
    SVD_test[np.abs(SVD_test) < 10e-12] = 0
    print(SVD_test)

    v_rows, rank = V_T.shape[0], A.shape[1]
    cosines = []
    doc_matrix = np.zeros((v_rows, rank))
    q_hat = np.zeros(rank)
    identity = np.eye(rank)
    # doc_matrix = get_document_matrix(Sigma, V_T)
    for k in range(rank):
        '''doc_matrix[:,k] = Sigma[k]*V_T[:,k]
        q_hat[k] = np.dot(U[:,k], q)
        cosines[k] = np.dot'''
        D = get_document_matrix(Sigma, V_T, k+1)
        print(f'shape of D: {D.shape}')
        qk = get_query(U, q, k+1)
        print(f'shape of qk: {qk.shape}')
        denom = np.linalg.norm(qk)*np.linalg.norm(D[:,k] @ identity[:,k])
        cosines.append(np.dot(qk, D[:,k] @ identity[:,k])/denom)
        print(f'cosine for rank {k+1}: {cosines[k]}')

        