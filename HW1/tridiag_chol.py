import numpy as np
from time import perf_counter_ns



def getTridiagonalInverse(subD, mainD, supD):
    # great speedup - still room for improvement - play with indices later to avoid separate loops
    n = len(mainD)
    theta = np.ones(n+1)
    psi = np.ones(n+1)
    theta[1] = mainD[0]
    psi[-2] = mainD[-1]
    for i in range(2,n+1):
        theta[i] = mainD[i-1]*theta[i-1] - supD[i-2]*subD[i-2]*theta[i-2]
    for i in range(n-2,-1,-1):
        psi[i] = mainD[i]*psi[i+1] - supD[i]*subD[i]*psi[i+2]
    Ainv = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i < j:
                Ainv[i,j] = (((-1)**(i+j))*np.prod(supD[i:j])*theta[i]*psi[j+1])/theta[n]
            elif i > j:
                Ainv[i,j] = (((-1)**(i+j))*np.prod(subD[j:i])*theta[j]*psi[i+1])/theta[n]
            else:
                Ainv[i,j] = (theta[i]*psi[j+1])/theta[n]
    return Ainv


def testInverse(A, inv):
    if not (np.allclose(np.linalg.inv(A), inv)):
        raise Exception(f'Something went wrong with new matrix inverse method')


def getTriDiagChol(subD, mainD, supD):
    n = len(mainD)
    A = np.zeros((n,n))
    diags = np.multiply(subD, supD)
    np.fill_diagonal(A, np.square(mainD))
    # works because fill_diagonal is in-place
    np.fill_diagonal(A[:(n-1), 1:], diags)
    np.fill_diagonal(A[1:, :(n-1)], diags)
    return A


def getMatrix(subD, mainD, supD):
    n = len(mainD)
    A = np.zeros((n,n))
    np.fill_diagonal(A, mainD)
    # works because fill_diagonal is in-place
    np.fill_diagonal(A[:(n-1), 1:], supD)
    np.fill_diagonal(A[1:, :(n-1)], subD)
    return A


if __name__ == "__main__":
    subDiag = -1*np.ones(9)
    supDiag = -1*np.ones(9)
    mainDiag = 2*np.ones(10)
    B = getMatrix(subDiag, mainDiag, supDiag)
    new_inv = getTridiagonalInverse(subDiag, mainDiag, supDiag)
    testInverse(B, new_inv)
    cholFactor = getTriDiagChol(subDiag, mainDiag, supDiag)
    # a. Find condition number κ2(B)
    A_norm = np.sqrt(max(np.linalg.eigvals(cholFactor)))
    Ainv_norm = np.sqrt(max(np.linalg.eigvals(np.matmul(new_inv.T,new_inv))))
    condNumber = A_norm*Ainv_norm
    print(f'condition number κ2(B): {condNumber}')
    # b. Find the smallest (λmin) and the largest eigenvalues (λmax) of B to
    #compute the ratio λmax/λmin.
    eigenvals = np.linalg.eigvals(B)
    eigenvalRatio = max(eigenvals)/min(eigenvals)
    print(f'ratio λmax/λmin: {eigenvalRatio}')
    # c. compare the above results
    print('conclusion: ratio > condition number')