import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

# a class is probably unnecessary but I'm fitting a lot into this one file
class IDD_test(object):
    def __init__(self, A):
        self.A = A
        self.A_shape = A.shape

    def make_dir_graph(self):
        rows, cols = np.where(self.A != 0)
        edges = zip(list(rows), list(cols))
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(edges)
    
    def draw_graph(self):
        if not hasattr(self, 'graph'):
            self.make_dir_graph()
        node_labels = dict(enumerate([chr(65+i) for i in range(self.A_shape[0])]))
        nx.draw(self.graph, node_size=500, labels = node_labels)
        plt.show()

    def is_reducible(self):
        self.make_dir_graph()
        return nx.is_strongly_connected(self.graph)

    def is_diag_dominant(self):
        m = self.A.shape[0]
        Lambda = np.zeros(m)
        dominance_test = np.zeros(m, dtype=bool)
        strictness_test = np.zeros(m, dtype=bool)
        for i in range(m):
            Lambda[i] = np.linalg.norm(self.A[i], ord=0) - np.abs(self.A[i,i])
            dominance_test[i] = (np.abs(self.A[i,i]) >= Lambda[i])
            strictness_test[i] = (np.abs(self.A[i,i]) > Lambda[i])
        return (sum(dominance_test) == m) and (sum(strictness_test) > 0)

    def is_IDD(self):
        return self.is_reducible() and self.is_diag_dominant()

class RelaxationMethods(object):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.A_shape = A.shape
        if (self.A_shape[0] != self.A_shape[1]): 
            raise Exception(f'ERROR: A is non-square with dimensions {self.A_shape[0]}x{self.A_shape[1]}! \nRelaxation methods require square matrices.')
        if (len(self.A_shape) != 2):
            raise Exception(f'ERROR: A is of incorrect shape {self.A_shape}! \nRelaxation methods require two-dimensional matrices.')
        self.D = np.diag(np.diag(A))
        self.E = -np.tril(A)
        self.F = -np.triu(A)
        self.relax_methods = ['Jacobi', 'GaussSeidel', 'SOR']

    def relaxation(self, init_x, method='Jacobi', omega=None, max_iter=100, TOL=10e-8):
        if method not in self.relax_methods:
            raise Exception(f'ERROR: unspecified method \'{method}\' given in call to relaxation! \nChoose from {self.relax_methods}')
        x = [init_x]
        k = 0
        stop_flag = False
        while (k < max_iter) and not stop_flag:
            if (k > 0 and np.linalg.norm(x[k]-x[k-1], ord=np.inf) < TOL):
                stop_flag = True
                break
            if omega is None:
                x.append(getattr(self, method)(x[k]))
            else:
                x.append(getattr(self, method)(x[k], omega))
            k += 1
        return np.array(x)

    def get_spectral_radius(self, method='Jacobi', omega=None):
        if method not in self.relax_methods:
            raise Exception(f'ERROR: unspecified method \'{method}\' given in call to relaxation! \nChoose from {self.relax_methods}')
        # spectral radius function
        def rho(T):
            eigvals, eigvec = np.linalg.eig(T)
            return(max(np.abs(eigvals)))

        if method == 'Jacobi':
            return rho(np.linalg.inv(self.D) @ (self.E + self.F))
        elif method == 'GaussSeidel':
            return rho(np.linalg.inv(self.D-self.E) @ self.F)
        elif method == 'SOR':
            return rho(np.linalg.inv(self.D-omega*self.E) @ ((1-omega)*self.D + omega*self.F))

    def Jacobi(self, x_prev):
        xk = np.zeros(self.A_shape[1])
        for i in range(self.A_shape[1]):
            xk[i] = (1/self.A[i,i])*(self.b[i] - np.dot(self.A[i,:i], x_prev[:i]) - np.dot(self.A[i,(i+1):], x_prev[(i+1):]))
        return xk

    def GaussSeidel(self, x_prev):
        xk = np.zeros(self.A_shape[1])
        for i in range(self.A_shape[1]):
            xk[i] = (1/self.A[i,i])*(self.b[i] - np.dot(self.A[i,:i], xk[:i]) - np.dot(self.A[i,(i+1):], x_prev[(i+1):]))
        return xk

    def SOR(self, x_prev, omega):
        if (omega < 0) or (omega > 2):
            raise Exception('ERROR: invalid value {omega} given for omega. \nomega must be in the open interval (0,2)')
        x_GS = self.GaussSeidel(x_prev)
        return (1-omega)*x_prev + omega*x_GS

    def get_optimal_omega(self, x0, exact_sol, iterations=10):
        radius = self.get_spectral_radius(method='Jacobi')
        if (radius > 0) and (radius < 1):
            omega = (1 + 0.25*(radius**2) + 0.125*(radius**4))
            test_sol = self.relaxation(x0, method='SOR', omega=omega, max_iter=iterations)[-1]
            return omega, test_sol
        else:
            return self.omega_bin_search(x0, exact_sol)

    # binary search for solution
        # pretty computationally expensive overall - if spectral radius of iteration matrix in [0,1], use theoretical optimal formula
    def omega_bin_search(self, x0, exact_sol, iterations=10, min_int_size=10e-4):
        interval = [0,2]
        solutions = [] # solutions found for each omega
        # while the size of the most recent subinterval is less than some TOL
        while(np.abs(interval[1]-interval[0]) > min_int_size):
            # pivot point
            mid_point = (interval[1]+interval[0])/2
            # subintervals separated by pivot point
            subintervals = [[interval[0], mid_point], [mid_point, interval[1]]]
            # random values chosen uniformly on each subinterval and corresponding solutions
            test_omega = [np.random.uniform(*subintervals[0]), np.random.uniform(*subintervals[1])]
            test_sol = [self.relaxation(x0, method='SOR', omega=test_omega[0], max_iter=iterations)[-1],
                        self.relaxation(x0, method='SOR', omega=test_omega[1], max_iter=iterations)[-1]]
            # compare error in each test solution then get index of smallest error
            errors = [np.linalg.norm(exact_sol-sol) for sol in test_sol]
            min_idx = np.argmin(errors)
            # save omega and corresponding solution that gave smallest error
            solutions.append((test_omega[min_idx], test_sol[min_idx]))
            # update interval to be the subinterval that produced the smallest error
            interval = subintervals[min_idx]
        # return most recent solution
        return solutions[-1]


# using Example 6.17 in the lecture note as a ground truth
def test_Jacobi():
    A = np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
    b = np.array([1,0,5])
    x0 = np.array([1,1,1])
    test_obj = RelaxationMethods(A, b)
    # assert np.allclose(test_obj.relaxation(x0, max_iter=3)[-1], [3/2, 2, 7/2])

if __name__ == '__main__':
    A = np.array([[2,-2,0,0],
                [-1,2,-1,0],
                [0,-1,2,-1],
                [0,0,-1,2]])
    b = np.array([-2,-2,-2,14])
    exact_sol = np.array([0,1,4,9])
    x0 = np.array([0,0,0,0])
    #################################################################################################
    # part (a) - determine if A is irreducibly diagonally-dominant
    truth_list = ['is not', 'is']
    IID_obj = IDD_test(A)
    IID_obj.draw_graph()
    print(f'A {truth_list[int(IID_obj.is_IDD())]} irreducibly diagonally-dominant.\n\n')
    #################################################################################################
    # part (b) - perform 10 iterations of Jacobi and Gauss-Seidel Methods at x0
    relax_obj = RelaxationMethods(A, b)
    Jacobi_sol = relax_obj.relaxation(x0, method='Jacobi', max_iter=10)
    GS_sol = relax_obj.relaxation(x0, method='GaussSeidel', max_iter=10)
    print(f'10 iterations of Jacobi method:\n {Jacobi_sol}')
    print(f'10 iterations of Gauss-Seidel method:\n {GS_sol}\n\n')
    #################################################################################################
    # part (c) - find best omega and compare with random omega
    best_omega, best_sol = relax_obj.get_optimal_omega(x0, exact_sol, iterations=10)
    print(f'optimal omega found by either formula or binary search: {best_omega}')
    print(f'corresponding solution to optimal omega: \n{best_sol}')
    print(f'infinity norm of this solution: {np.linalg.norm(best_sol-exact_sol, ord=np.inf)}')
    random_omega = np.random.uniform(0,2)
    rand_SOR_sol = relax_obj.relaxation(x0, method='SOR', omega = random_omega, max_iter=10)
    print(f'random omega from uniform interval (0,2): {random_omega}')
    print(f'corresponding solution to random omega: \n{rand_SOR_sol[-1]}')
    print(f'infinity norm of this solution: {np.linalg.norm(rand_SOR_sol[-1]-exact_sol, ord=np.inf)}\n\n')
    #################################################################################################
    # part (d) - display spectral radii of iteration matrices for each method
    print('spectral radius of Jacobi iteration matrix: %f' % relax_obj.get_spectral_radius(method='Jacobi'))
    print('spectral radius of Gauss-Seidel iteration matrix: %f' % relax_obj.get_spectral_radius(method='GaussSeidel'))
    print('spectral radius of SOR iteration matrix: %f' % relax_obj.get_spectral_radius(method='SOR', omega=best_omega))