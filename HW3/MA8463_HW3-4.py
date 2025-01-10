# Author: Jacob Kutch
# Originally written in early 2020 shortly after starting to learn Python

import sys
import numpy as np
from statistics import mean, stdev
#from scipy import solve_triangular
#import matplotlib.pyplot as plt

def containsDuplicates(listInput) -> bool:
    # checks for duplicates in an array of integers using hashing with O(n) complexity and space req
    data = set()
    for i in listInput:
        if i in data:
            return True # stops searching upon first duplicate found
        data.add(i) # adds elements to the set on each pass
    return False

# reduced features and removed some notes from the original for brevity
class LSRegression:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.xdata = x
        self.ydata = y

    def polyEval(self, xx: np.ndarray) -> np.ndarray: # xx contains arguments in which to evaluate the polynomial
        # returns y-values at these points in xx
        '''if !(hasattr(self, 'coef')):
            self.coef = self.polyRegression()'''
        yy = np.zeros(len(xx)) # to store approximations
        for i in range(len(self.coef)):
            yy += self.coef[i]*(xx**i) # adds Ci*x**i pairs to yy one term (of the polynomial) at a time
        return yy

    def polyRegression(self, deg: int): # in least squares sense
        # solves the system of normal equations V'Va = V'y where V is the Vandermonde matrix of x, V' is its transpose, and a contains coefficients
            # of the approximated polynomial - often simplified to Sa = b where S = V'V and b = V'y
            # reference found here: https://mathworld.wolfram.com/LeastSquaresFittingPolynomial.html
        if len(self.xdata) != len(self.ydata):
            print("arrays of input and output data must be the same length!")
            sys.exit(1)
        if(containsDuplicates(self.xdata)):
            print("x values in input data must be distinct!")
            sys.exit(1)

        size = len(self.xdata)
        s = np.zeros(2*deg+1) # representing row elements of S = V'V
        b = np.zeros(deg+1)     # elements of b = V'y
        Smatrix = np.zeros((deg+1,deg+1)) # preallocate S
        for k in range(len(s)):
            for i in range(size):
                if k <= deg:
                    b[k] += (self.xdata[i]**k)*(self.ydata[i])
                s[k] += self.xdata[i]**k
        # filling S like so is easier than using the Vandermonde matrix and its transpose
        for i in range(deg+1):
            Smatrix[i] = s[i:(deg+i+1)] # increments along rows and columns so that skew diagonals of S are filled with s[1]  
        self.coef = np.linalg.solve(Smatrix, b) # solving for the a values in the augmented matrix Sa = b
        # coefficients stored in lowest degree order first, i.e., [c0,c1,c2,...] for c0 + c1*x + c2*x**2 + ...

# I've rewritten this a half dozen times and am now electing to follow Wikipedia exactly: https://en.wikipedia.org/wiki/QR_decomposition
def classicalGramSchmidt(A):
    m, n = np.shape(A)
    R = np.zeros((n,n))
    U = np.zeros((m,n))
    Q = np.zeros((m,n))
    projection = lambda a,b : (np.dot(a,b)/np.dot(b,b))*b
    # loop finds each u_i
    for cols in range(n):
        summation = np.zeros(len(A[:,cols]))
        for j in range(cols):
            summation += projection(A[:,cols],U[:,j])
        U[:,cols] = A[:,cols] - summation
        Q[:,cols] = U[:,cols]/np.linalg.norm(U[:,cols])
        for rows in range(cols+1):
            R[rows,cols] = np.dot(Q[:,rows],A[:,cols])
    return Q, R

if __name__ == '__main__':
    np.set_printoptions(16)
    # Part (a)
    x = np.array(range(1,6))
    y = np.array([0.8,2.1,3.3,4.1,4.7])
    A = np.stack((np.ones(len(x)), x), axis=-1)

    # Part (b)(i)
    model = LSRegression(x,y)
    model.polyRegression(1)
    x_hat1 = model.coef
    print(f'Least Squares Coefficients:\n {x_hat1}')
    '''xx = np.linspace(0,6,100)
    plt.scatter(x,y)
    plt.plot(xx, model.polyEval(xx))
    plt.show()'''

    # Part (b)(ii)
    Q, R = classicalGramSchmidt(A)
    print(f'computed Q =\n {Q}')
    print(f'computed R =\n {R}')
    print(f'QR = {np.matmul(Q,R)}')
    print(f'A = QR:\n {np.allclose(A, np.matmul(Q,R))}')
    x_hat2 = np.linalg.solve(R, np.matmul(Q.T,y))
    print(f'QR Decomposition Coefficients:\n {x_hat2}')
    '''Q, R = np.linalg.qr(A)
    print(f'true Q = {Q}')
    print(f'true R = {R}')'''
    #sol = np.linalg.solve(np.matmul(A.T,A),np.matmul(A.T,y))

    # Part (c)
    g = lambda x, a0, a1: a0 + a1*x
    LS_residuals = np.abs(y - g(x, x_hat1[0], x_hat1[1]))
    QR_residuals = np.abs(y - g(x, x_hat2[0], x_hat2[1]))
    print(f'Least Squares Error:\n {LS_residuals}')
    print(f'QR Decomposition Error:\n {QR_residuals}')

    print(np.linalg.norm(LS_residuals))
    print(np.linalg.norm(QR_residuals))