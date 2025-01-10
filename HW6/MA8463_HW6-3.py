import numpy as np
from scipy.sparse import diags

# 6.3 in HW 6
def get_A_b(interval: tuple, n: int, f, u) -> tuple:
    A = np.zeros((n+1,n+1))
    b = np.zeros(n+1)
    h = (interval[1]-interval[0])/n
    for i in range(1,n):
        A[i,i-1], A[i, i+1] = -1, -1
        A[i,i] = 2
        b[i] = (h**2)*f(interval[0]+i*h)
    A[0,0], A[n,n] = 1, 1
    b[0], b[n] = u(interval[0]), u(interval[1])
    return A, b


# code below more heavily modified from lecture notes
if __name__ == '__main__':
    # BVP in ex. 6.3.9
    u = lambda x: np.sin(np.pi*x) + 2*x
    f = lambda x: (np.pi**2)*np.sin(np.pi*x)
    interval = [0,1]
    BVs = [0,2]
    n = np.array([int(10*(2**i)) for i in range(3)]) # n0 = 10 in HW
    h = 1/n
    print(h)
    U, V, W, WT = [], [], [], []
    for i in range(3):
        #n.append(int(n0*(2**(i-1))))
        X = np.linspace(interval[0], interval[1], n[i]+1)
        U.append(u(X))
        A, b = get_A_b(interval, n[i], f, u)
        V.append(np.linalg.solve(A,b))
    for i in range(2):
        # Richardson extrapolation on the course mesh
        W.append((1/3)*(4*V[i+1][:len(V[i+1])+1:2]-V[i])) # 1 on al 6.39
    # 6th order solution on the mid mesh
    W.append((1/15)*(16*W[1][:len(W[1])+1:2]-W[0]))
    WT.append(W[0]) 
    WT[0] = W[:len(W[0])+1:2]
    # get a sixth order solution on the mid mesh for example 6.36 and algorithm 6.39
    # print(np.linalg.norm(W[2]-U))
    print(n)
    for i in range(len(W)):
        print(f'W_{i} : {np.shape(W[i])}')
        print(f'U_{i} : {np.shape(U[i])}')
        print(f'V_{i} : {np.shape(V[i])}')
    print(f'X : {np.shape(X)}')
    print(W[2]-f(X))
