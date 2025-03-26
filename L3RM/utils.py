import numpy as np

def first_order_derivative_filter(n,k=1):
    return np.eye(n, k=k)-np.eye(n)

def get_matrix_gradient(matrix: np.ndarray, 
                        dx: np.ndarray, 
                        dy: np.ndarray):
    delx = matrix @ dx
    dely = dy @ matrix

    return delx, dely