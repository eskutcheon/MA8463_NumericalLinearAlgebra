import numpy as np

# reference was Wikipedia: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability
def modifiedGramSchmidt(A):
    m, n = np.shape(A)
    R = np.zeros((n,n))
    U = np.zeros((m,n))
    Q = np.zeros((m,n))
    projection = lambda a,b: (np.dot(a,b)/np.dot(b,b))*b
    for cols in range(n):
        U[:,cols] = A[:,cols] - (cols!=0)*projection(A[:,cols],A[:,0])
        for k in range(1,cols):
            U[:,cols] -= projection(U[:,cols],U[:,k])
        Q[:,cols] = U[:,cols]/np.linalg.norm(U[:,cols])
        for rows in range(cols+1):
            R[rows,cols] = np.dot(Q[:,rows],A[:,cols])
    return Q, R

def orthogonalIter(A, Z0, max_iter = 50, TOL=10e-8):
    Z, Y = [Z0], []
    i = 0
    flag = False
    while (not flag) and (i < max_iter):
        Y.append(A @ Z[i])
        Z.append(modifiedGramSchmidt(Y[i])[0])
        i += 1
        if np.allclose(Z[i], Z[i-1], rtol=TOL): flag = True
    return np.array(Z)


def testSubspace(A, Z):
    '''test if Z is an invariant subspace of A corresponding to the eigenspace of the largest eigenvalues
        Here, Z is the final iteration returned by the orthogonal iteration algorithm
    '''
    eigVal, eigVec = np.linalg.eig(A)
    # sort in order of largest singular values
    eigIndices = np.argsort(eigVal)[::-1]
    eigVec = eigVec[eigIndices]
    Z = Z[eigIndices]
    print(f'(sorted) eigenvectors of A:\n{eigVec}\n')
    print(f'(sorted) Z_k = \n{Z}\n')
    print(f'note: The absolute values of the elements in \n{eigVec[:,0]} and \n{Z[:,0]} \n should be roughly the same\n')
    testArr = np.column_stack((eigVec[:,0],Z[:,0]))
    # test if the first column of Z is in an eigenspace of A
        # could probably also use np.allclose
    if (np.linalg.matrix_rank(testArr) < (eigVec.shape[1]+1)):
        return True


if __name__ == '__main__':
    A = np.array([[-31,-35,16],[-10,-8,4],[-100,-104,49]])
    n = A.shape[0]
    spectrum = np.array([9,3,-2]) # eigenvalues of A in descending order
    Z0 = np.eye(n)[:,:2] # first 2 standard basis vectors
    Z = orthogonalIter(A, Z0, 20, 10e-6)
    truthLabels = ['is not', 'is']
    idx = testSubspace(A, Z[-1])
    print(f'Z_k {truthLabels[idx]} an invariant subspace of A.')
