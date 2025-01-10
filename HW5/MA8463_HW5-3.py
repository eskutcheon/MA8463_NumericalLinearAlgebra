import numpy as np

def invPowerIter(A, sigma, x0, max_iter = 50, TOL=10e-8):
    B = A - sigma*np.eye(np.shape(A)[0])
    i = 0
    x, y, eta = [x0], [], []
    flag = False
    while (not flag) and (i < max_iter):
        y.append(np.linalg.solve(B, x[i]))
        x.append(y[i]/np.linalg.norm(y[i]))
        eta.append(x[i+1].T @ (A @ x[i+1]))
        i += 1
        if (np.linalg.norm(x[i]-x[i-1]) < TOL): flag = True
    print(f'eigenvector for each iteration:\n{np.array(x)}')
    print(f'eigenvalue for each iteration:\n{np.array(eta)}\n')
    return np.array(x[1:]), np.array(eta)

def getEigVecConv(trueEigVec, appEigVec):
    numIter = len(appEigVec)
    absErr = np.zeros(numIter)
    convRates = np.zeros(numIter-1)
    for i in range(numIter):
        absErr[i] = np.linalg.norm(trueEigVec - appEigVec[i])
        if i < (numIter-1):
            convRates[i] = np.linalg.norm(trueEigVec - appEigVec[i+1])/absErr[i]
    print(f'absolute error in dominant eigenvector for each iteration:\n{absErr}')
    print(f'convergence rate in dominant eigenvector for each iteration:\n{convRates}\n')

def getEigValConv(trueEigVal, appEigVal):
    numIter = len(appEigVal)
    absErr = np.zeros(numIter)
    convRates = np.zeros(numIter-1)
    for i in range(numIter):
        absErr[i] = np.abs(trueEigVal-appEigVal[i])
        if i < (numIter-1):
            convRates[i] = np.abs(trueEigVal-appEigVal[i+1])/absErr[i]
    print(f'absolute error in dominant eigenvalue for each iteration:\n{absErr}')
    print(f'convergence rate in dominant eigenvalue for each iteration:\n{convRates}\n')

if __name__ == '__main__':
    A = np.array([[1,1,1],[-1,9,2],[0,-1,2]])
    shift = 9
    initGuess = np.array([1,1,1])
    eigenVec, eigenVal = invPowerIter(A, shift, initGuess, 10)
    trueEigVal, trueEigVec = np.linalg.eig(A)
    eigIndices = np.argsort(trueEigVal)
    print(f'true eigenvectors sorted in order of largest singular values:\n{trueEigVec[eigIndices][::-1]}')
    print(f'true eigenvalues sorted in order of largest singular values:\n{trueEigVal[eigIndices][::-1]}\n')
    domEigVal = trueEigVal[eigIndices][-1]
    domEigVec = trueEigVec[eigIndices][-1]
    getEigVecConv(domEigVec, eigenVec)
    getEigValConv(domEigVal, eigenVal)
