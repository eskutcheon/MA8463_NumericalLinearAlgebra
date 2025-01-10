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

def QRiter_noShift(A0, max_iter = 50, TOL = 10e-8):
    A = [A0]
    n = A0.shape[0]
    i = 0
    flag = False
    while (not flag) and (i < max_iter):
        Q, R = modifiedGramSchmidt(A[i])
        A.append(np.matmul(R, Q))
        i += 1
        if (np.abs(A[i][n-1][n-1]-A[i-1][n-1][n-1]) < TOL): flag = True
    return np.array(A)

def QRiter_withShift(A0, max_iter = 50, TOL = 10e-8):
    A = [A0]
    sigma = []
    n = A0.shape[0]
    i = 0
    flag = False
    while (not flag) and (i < max_iter):
        sigma.append(A[i][2][2]/2)
        diag = sigma[i]*np.eye(n)
        Q, R = modifiedGramSchmidt(A[i] - diag)
        A.append((R @ Q) + diag)
        i += 1
        if (np.abs(A[i][n-1][n-1]-A[i-1][n-1][n-1]) < TOL): flag = True
    return np.array(A)

def getConvRate(eigVals, A):
    numIter = A.shape[0]
    convRates = np.zeros(numIter-1)
    for i in range(numIter-1):
        appEigenVals_next = np.diag(A[i+1])
        appEigenVals_prev = np.diag(A[i])
        convRates[i] = np.linalg.norm(appEigenVals_next-eigVals)/np.linalg.norm(appEigenVals_prev-eigVals)
    return convRates

def testEigenVals(trueEigVals, appEigVals):
    truthLabels = ['are not', 'are']
    print(f'The approximated eigenvalues,\n{appEigVals} \n' +
          f'{truthLabels[np.allclose(trueEigVals, appEigVals)]}' +
          f' roughly the same as the true eigenvalues, \n{trueEigVals}')
    print(f'Error: {np.linalg.norm(trueEigVals - appEigVals)}\n')

if __name__ == '__main__':
    A0 = np.array([[-31,-35,16],[-10,-8,4],[-100,-104,49]])
    spectrum = np.array([9,3,-2])
    A_noShift = QRiter_noShift(A0, 30)
    appEigenVals = np.diag(A_noShift[-1])
    testEigenVals(spectrum, appEigenVals)
    A_shift = QRiter_withShift(A0, max_iter = 30)
    appEigenVals = np.sort(np.diag(A_shift[-1]), axis=None)[::-1]
    testEigenVals(spectrum, appEigenVals)

    noShiftConvRates = getConvRate(spectrum, A_noShift)
    # error looked much larger when passing spectrum and sorting withing the function
    shiftConvRates = getConvRate(np.array([9,-2,3]), A_shift)
    print(f'convergence rates with no shifting: v1 = \n{noShiftConvRates}\n')
    print(f'convergence rates with shifting: v2 = \n{shiftConvRates}\n')
    print(f'||v1|| = {np.linalg.norm(noShiftConvRates)}')
    print(f'||v2|| = {np.linalg.norm(shiftConvRates)}')