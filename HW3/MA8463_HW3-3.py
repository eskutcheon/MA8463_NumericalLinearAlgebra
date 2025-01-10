# comments are scarce because I'm tired of rewriting things at this point - will add before posting to Github eventually

import numpy as np
from copy import copy, deepcopy

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

# I've rewritten this a half dozen times and am now electing to follow Wikipedia exactly: https://en.wikipedia.org/wiki/QR_decomposition
def classicalGramSchmidt(A):
    m, n = np.shape(A)
    R = np.zeros((n,n))
    U = np.zeros((m,n))
    Q = np.zeros((m,n))
    projection = lambda a,b: (np.dot(a,b)/np.dot(b,b))*b
    for cols in range(n):
        summation = np.zeros(len(A[:,cols]))
        for j in range(cols):
            summation += projection(A[:,cols],U[:,j])
        U[:,cols] = A[:,cols] - summation
        Q[:,cols] = U[:,cols]/np.linalg.norm(U[:,cols])
        for rows in range(cols+1):
            R[rows,cols] = np.dot(Q[:,rows],A[:,cols])
    return Q, R

# Again, modifications were made by referencing Wikipedia: https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability
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
        R[j:,:] -= np.matmul(bvvt, R[j:,:])
        Q[:,j:] -= np.matmul(Q[:,j:], bvvt)
    return Q, np.triu(R[:m])

# only works for 2d case
def givens(a, b):
    if b == 0:
        return 1, 0
    else:
        if np.abs(b) >= np.abs(a):
            tau = -a/b
            s = 1/np.sqrt(1+tau**2)
            c = tau*s
        else:
            tau = -b/a
            c = 1/np.sqrt(1+tau**2)
            s = tau*c
        return c, s

def givensRotations(A):
    m, n = np.shape(A)
    R = -copy(A)
    Q = -np.eye(m)
    for j in range(n):
        for i in range(m-1,j,-1):
            c, s = givens(R[i-1,j], R[i,j])
            G = np.array([[c,-s], [s,c]]).T
            R[(i-1):(i+1),:] = np.matmul(G.T, R[(i-1):(i+1),:])
            Q[:,(i-1):(i+1)] = np.matmul(Q[:,(i-1):(i+1)], G)
    return Q, R

def testFactorization(A, Q, R):
    print(f'Q =\n {Q}')
    print(f'R =\n {R}')
    print(f'A =\n {A}')
    diag = np.diag(R)
    return np.allclose(np.matmul(Q,R), A) and np.allclose(np.matmul(Q.T,Q), np.eye(np.shape(Q)[1])) and len(diag[diag > 0]) == np.shape(R)[1]

if __name__ == '__main__':
    A = np.array([[1,1,0], [0,1,1], [-1,0,2], [1,1,1]], dtype=np.float64)
    Q_actual = np.array([[1/np.sqrt(3), 1/np.sqrt(15), -1/np.sqrt(35)],
                        [0, 3/np.sqrt(15), -3/np.sqrt(35)],
                        [-1/np.sqrt(3), 2/np.sqrt(15), 3/np.sqrt(35)],
                        [1/np.sqrt(3), 1/np.sqrt(15), 4/np.sqrt(35)]])
    R_actual = np.array([[np.sqrt(3), 2/np.sqrt(3), -1/np.sqrt(3)],
                        [0, 5/np.sqrt(15), 8/np.sqrt(15)],
                        [0, 0, 7/np.sqrt(35)]])
    truthList = ['do not', 'do']

    Q1, R1 = classicalGramSchmidt(A)
    print('Classical Gram-Schmidt:')
    print(f'Q and R {truthList[testFactorization(A,Q1,R1)]} give the QR Factorization of A.\n')

    Q2, R2 = modifiedGramSchmidt(A)
    print('Modified Gram-Schmidt:')
    print(f'Q and R {truthList[testFactorization(A,Q2,R2)]} give the QR Factorization of A.\n')

    Q3, R3 = householderReflection(A)
    print('Householder Reflection:')
    print(f'Q and R {truthList[testFactorization(A,Q3,R3)]} give the QR Factorization of A.\n')
    R3 = R3[:3]

    Q4, R4 = givensRotations(A)
    print('Givens Rotation:')
    print(f'Q and R {truthList[testFactorization(A,Q4,R4)]} give the QR Factorization of A.\n')