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

def get_W_expansion(W, f, h, n):
    W_hat = np.zeros(n+1)
    W_hat[0:(n+1):2] = W
    for k in range(1,n,2): # might be a problem with indices if this was supposed to be to n+1
        W_hat[k] = 0.5*(W_hat[k-1]+W_hat[k+1]) + ((h**2)/24)*(f((k-1)*h) + 10*f(k*h) + f((k+1)*h))
    return W_hat

# get sixth order solution up to arbitrarily fine meshes
def get_sixth_approx(W_hat, mesh_idx, n, u_exact):
    solution = ((1/15)*(16*W_hat[mesh_idx-1][:(n[mesh_idx]+1):2] - W_hat[mesh_idx-2]))
    error = np.linalg.norm(u_exact-solution, ord=np.inf)
    return solution, error

def get_mesh_names(num):
    name_list = ['coarse', 'mid', 'fine']
    if num > 3:
        for _ in range(num-3):
            name_list.append(f'super_%s' % name_list[-1])
    return name_list

def Richardson_Extrapolation(u, f, interval, num_meshes):
    # partitioning into subintervals
    # number of subintervals for mesh i, with 10 initial points on BVP interval
    n = np.array([int(10*(2**i)) for i in range(num_meshes)])
    h = 1/n
    # generate spacial grid points for 
    X = [np.linspace(interval[0], interval[1], n[i]+1) for i in range(num_meshes)]
    # preallocate u and approximations on the meshes
    U = [np.zeros(n[i]+1) for i in range(num_meshes)]
    V = [np.zeros(n[i]+1) for i in range(num_meshes)]
    W = [np.zeros(n[i]+1) for i in range(num_meshes)]
    W_hat = [np.zeros(n[i]+1) for i in range(1,num_meshes)]

    for i in range(num_meshes):
        U[i] = u(X[i]) # exact solutions to test error later
        A, b = get_A_b(interval, n[i], f, u) # get tridiagonal matrix and b = (k^2)*f_i where k = [h, h/2, h/4]
        V[i] = np.linalg.solve(A,b) # solve for v_h, v_(h/2), and v_(h/4)
        # Perform Richardson Extrapolation
        if (i > 0) and (i < num_meshes):
            W[i-1] = (1/3)*(4*V[i][:len(V[i])+1:2]-V[i-1]) # 1 on algorithm 6.39
            W_hat[i-1] = get_W_expansion(W[i-1], f, h[i], n[i])
    return get_sixth_approx(W_hat, num_meshes-1, n, U[num_meshes-2])

if __name__ == '__main__':
    # BVP in ex. 6.3.9
    u = lambda x: np.sin(np.pi*x) + 2*x
    f = lambda x: (np.pi**2)*np.sin(np.pi*x)
    interval = [0,1]
    BVs = [0,2]

    #################################################################################
    # mesh generation section

    for num_meshes in range(3,8):
        meshes = get_mesh_names(num_meshes)
        solution, error = Richardson_Extrapolation(u, f, interval, num_meshes)
        # solution, error = get_sixth_approx(W_hat, num_meshes-1, n, U[num_meshes-2])
        # print(f'6th order approximation of u on the {meshes[num_meshes-2]} mesh: \n{solution}\n')
        print(f'error (inf norm) of approximation : \n{error}')
