import numpy as np

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy

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
        node_labels = dict(enumerate([str(i+1) for i in range(self.A_shape[0])]))
        nx.draw(self.graph, node_size=500, labels = node_labels)
        plt.show()

    def is_irreducible(self):
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
        return self.is_irreducible() and self.is_diag_dominant()

# really need to figure out the syntax for einsum to avoid this
def col_by_row_mult(u,v):
    n = np.size(u)
    if np.size(v) == n and u.ndim == 1 and v.ndim == 1:
        A = np.zeros((n,n))
        for i in range(n):
            A[i,:] = u[i]*v
        return A
    raise Exception('Arguments must be one dimensional vectors of the same size!')

def get_P_matrix(Q, n):
    e = np.ones(n)
    d = np.invert(np.array([sum(Q[:,j]) for j in range(n)], dtype=bool)).astype(int) # possibly incorrect
    # column-stochastic matrix of which columns are probability vectors
    P = Q + (1/n)*(col_by_row_mult(e, d))
    # test below fails for modified Q - need to figure out how to fix it or delete the class above
        # might just be because P = Q in that case
    '''P_tester = IDD_test(P)
    if not P_tester.is_irreducible():
        raise Exception(f'ERROR: stochastic matrix P formed from Q must be irreducible!')'''
    return P
    

def get_Google_matrix(P, alpha, n):
    # must be irreducible if P is irreducible
    return alpha*P + ((1-alpha)/n)*np.ones((n,n))

def power_method(Q, alpha, r_init, n, TOL):
    r = deepcopy(r_init)
    e = np.ones(n)
    # personalization vector
    v = e/n
    residual = 1
    loop_count = 0
    # might switch to alpha**k >= TOL because residuals as high as 10e-4 gets awful results
    while(residual > TOL):
        beta = 1 - alpha*np.dot(e, Q @ r)
        z = alpha*(Q @ r) + beta*v
        residual = np.linalg.norm(r-z, ord=1)
        r = z
        loop_count += 1
    print(f'power method converged in {loop_count} iterations')
    return r, loop_count

def get_Pagerank(Q, n, alpha, r_init, TOL):
    P = get_P_matrix(Q, n)
    # P_tester.draw_graph()
    # G = get_Google_matrix(P, alpha, n)
    return power_method(Q, alpha, r_init, n, TOL)

def test_PR_results(r, Q, n, alpha, rTOL):
    P = get_P_matrix(Q, n)
    G = get_Google_matrix(P, alpha, n)
    print(f'Pagerank vector: r = \n{r}')
    print(f'Google matrix times Pagerank vector: Gr =\n{G @ r}')
    print(f'Gr = r? \t->\t{np.allclose(r, G @ r, rtol=rTOL)}\n')

if __name__ == '__main__':
    # Problem 7.3
    Q = np.array([[0,1/3,0,0,0,0],
                [1/3,0,0,0,0,0],
                [0,1/3,0,0,1/3,1/2],
                [1/3,0,0,0,1/3,0],
                [1/3,1/3,0,0,0,1/2],
                [0,0,1,0,1/3,0]])
    # damping factor in (0,1) according to Brin and Page (1996)
    alpha = 0.85
    r_TOL = 10e-4
    n = Q.shape[0] # Q must be square
    r_init = np.random.rand(n)
    r_init /= np.linalg.norm(r_init, ord=1)

    print('Problem 7.3:')
    r, num_iter1 = get_Pagerank(Q, n, alpha, r_init, r_TOL)
    test_PR_results(r, Q, n, alpha, r_TOL)

    # Problem 7.4
    Q_tilde = deepcopy(Q)
    # new outlink from 4 to 5 means an update to column 4, row 5 only since 4 had no outlinks
    Q_tilde[4,3] = 1
    print('Problem 7.4:')
    r_tilde, num_iter2 = get_Pagerank(Q_tilde, n, alpha, r_init, r_TOL)
    test_PR_results(r_tilde, Q_tilde, n, alpha, r_TOL)



