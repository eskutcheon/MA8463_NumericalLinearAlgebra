import numpy as np
from typing import Callable, Any
from time import perf_counter_ns
from scipy.sparse import linalg, diags


def getDEmatrix(f: Callable[[Any], Any], g: np.ndarray,
                a: np.ndarray, b: np.ndarray, c: np.ndarray,
                interval: tuple, n: int
) -> tuple:
    '''
        f := RHS of DE
        g := RHS of BCs
        a := coefficients of u in the BCs
        b := coefficients of ux in the BCs
        c := coefficients of the DE in the order uxx, ux, u
        interval := endpoint of the boundary [a,b]
        n := number of subintervals on the boundary
    '''
    h = (interval[1]-interval[0])/n
    A = np.zeros((n+1,n+1))
    # giving variable names for clarity
    mainDiag = -4*c[0] + 2*c[2]*h**2
    subDiag = 2*c[0] - c[1]*h
    superDiag = 2*c[0] + c[1]*h
    # fill tridiagonal matrix A in-place, skipping the first and last rows
    np.fill_diagonal(A[1:n,1:n], mainDiag)
    np.fill_diagonal(A[1:n, :(n-1)], subDiag)
    np.fill_diagonal(A[1:n, 2:], superDiag)
    # replacing first row using formula accounting for ghost grid value u_{-1}
    A[0, :3] = 4*a[0]*(c[0]*h - c[1]*h**2) + 2*b[0]*(c[2]*h**2 - 2*c[0]),
    4*b[0]*c[0], 0
    # replacing first row using formula accounting for ghost grid value u_{n+1}
    A[n, (n-2):] = 0, 4*b[1]*c[0], b[1]*(-4*c[0] + 2*c[2]*h**2) -
    2*a[1]*(2*c[0]*h + c[1]*h**2)
    RHS = np.zeros(n+1)
    xrange = np.linspace(interval[0], interval[1], n+1)[1:n]
    RHS[1:n] = (2*h**2)*f(xrange)
    # RHS using formulas accounting for ghost grid value u_{-1}
    RHS[0] = 2*f(interval[0])*b[0]*h**2 + 2*g[0]*(2*c[0]*h - c[1]*h**2)
    # RHS using formulas accounting for ghost grid value u_{n+1}
    RHS[n] = 2*(b[1]*f(interval[1])*h**2 - 2*c[0]*h*g[1] - c[1]*g[1]*h**2)
    return A, RHS



def getSparseMatrix(A: np.ndarray) -> np.ndarray:
    diagonals = [np.diag(A, k=-1), np.diag(A, k=0), np.diag(A, k=1)]
    diagOrder = [-1,0,1]
    return diags(diagonals, diagOrder)

def solveTimer(A : np.ndarray, b: np.ndarray, solver: Callable[[Any], Any]) -> tuple:
    startTime = perf_counter_ns()
    u = solver(A, b)
    endTime = perf_counter_ns()
    return u, (endTime-startTime)


if __name__ == "__main__":
    # for the specific BVP given in the homework
    f = lambda x : (np.pi**2 + 1)*np.cos(np.pi*x)
    interval = (0,1)
    BCs = [[1,0], [0,1]]
    coefs = [-1,0,1]
    BC_RHS = [1,0]
    # for each n given in the homework plus 3 more tests
    n = [25,50,100,200,500,1000]
    directSolveTimes = np.zeros(len(n))
    sparseSolveTimes = np.zeros(len(n))
    for i in range(len(n)):
        A, b = getDEmatrix(f, BC_RHS, BCs[0], BCs[1], coefs, interval, n[i])
        u, directSolveTimes[i] = solveTimer(A, b, np.linalg.solve)
        # print(f'approximation: u = {u}')
        xrange = np.linspace(interval[0],interval[1],n[i]+1)
        #print(f'true values: {np.cos(np.pi*xrange)}')
        u_error = np.abs(u - np.cos(np.pi*xrange))
        print(f'infinity-norm of residuals for n = {n[i]}: {max(u_error)}')
        A_sparse = getSparseMatrix(A)
        u_sparse, sparseSolveTimes[i] = solveTimer(A_sparse, b, linalg.spsolve)
        if not np.allclose(u, u_sparse):
            raise Exception('Solutions for direct solver and sparse solver are not the same')
    print(f'times for direct solver in nanoseconds: {directSolveTimes}')
    print(f'times for sparse solver in nanoseconds: {sparseSolveTimes}')