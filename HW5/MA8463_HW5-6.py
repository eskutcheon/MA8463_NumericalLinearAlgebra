import numpy as np
from copy import deepcopy

# because numpy.matmul apparently defaults to np.dot for 1d arrays
def colByRowMult(u, v):
    n = np.size(u)
    if np.size(v) == n and u.ndim == 1 and v.ndim == 1:
        A = np.zeros((n,n))
        for i in range(n):
            A[i,:] = u[i]*v
        return A
    else:
        raise Exception('Arguments must be one dimensional vectors of the same size')

# reference was Wikipedia: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability
def householder(x):
    n = len(x)
    sigma = np.dot(x[1:], x[1:])
    v = deepcopy(x)
    v[0] = 1
    beta = 0
    if sigma != 0:
        mu = np.sqrt(x[0]**2 + sigma)
        v[0] = x[0]-mu if x[0] <= 0 else -sigma/(x[0]+mu)
        beta = 2*(v[0]**2)/(sigma+v[0]**2)
        v /= v[0]
    return v, beta

def householderReflection(A):
    m, n = np.shape(A)
    R = deepcopy(A)
    Q = np.eye(m)
    for j in range(n):
        v, beta = householder(R[j:, j])
        bvvt = beta*colByRowMult(v, v)
        R[j:,:] -= (bvvt @ R[j:,:])
        Q[:,j:] -= (Q[:,j:] @ bvvt)
    return Q, np.triu(R[:m])

def HessReduction(A):
    n = max(np.shape(A))
    H = deepcopy(A)
    Q = np.eye(n)
    for k in range(n-2):
        v, beta = householder(H[(k+1):,k])
        bvvt = beta*colByRowMult(v, v)
        H[(k+1):,k:] -= (bvvt @ H[(k+1):,k:])
        H[:,(k+1):] -= (H[:,(k+1):] @ bvvt)
        Q[(k+1):,k:] -= (bvvt @ Q[(k+1):,k:])
        '''zeroMat = np.zeros((k,n-k))
        P = np.block([[np.eye(k), zeroMat],
                [zeroMat.T, np.eye(len(v))-bvvt]])
        print(P)
        Q = Q*P'''
    return Q, H

def singleShift_QR_Iter(A, init_shift, TOL = 10e-12, max_iter=100):
    Q0, H1 = HessReduction(A)
    n = A.shape[0]
    Qk = Q0
    H = [H1]
    k = 0
    flag = False
    mu = init_shift
    while (not flag) and (k < max_iter):
        shiftMatrix = mu*np.eye(n)
        Qk, Rk = householderReflection(H[k]-shiftMatrix)
        H.append((Rk @ Qk) + shiftMatrix)
        mu = H[k+1][-1][-1]
        k += 1
        if (np.allclose(H[k-1], H[k], atol=TOL)): flag = True
    return np.array(H)

def testEigenVals(trueEigVals, appEigVals):
    truthLabels = ['are not', 'are']
    isEqual = np.allclose(trueEigVals, appEigVals, atol=10e-12)
    print(f'The approximated eigenvalues,\n{appEigVals} \n' +
          f'{truthLabels[np.allclose(trueEigVals, appEigVals)]}' +
          f' roughly the same as the true eigenvalues, \n{trueEigVals}')
    print(f'Error: {np.linalg.norm(trueEigVals - appEigVals)}')

if __name__ == '__main__':
    A = np.array([[1,6,11,16,21],
                  [2,7,12,17,22],
                  [0,8,13,18,23],
                  [0,0,14,19,24],
                  [0,0,0,20,1]], dtype=np.float64)
    eigenVals, eigenVecs = np.linalg.eig(A)
    '''Q, H = HessReduction(A)
    print(H)
    print(Q)
    print((Q.T)@A@Q)'''
    H = singleShift_QR_Iter(A, init_shift=1, max_iter=30)
    finalH = H[-1]
    appEigVals = np.sort(np.diag(finalH))[::-1]
    testEigenVals(eigenVals, appEigVals)