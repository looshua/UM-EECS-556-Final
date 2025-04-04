import numpy as np
from scipy import sparse

def first_order_derivative_filter(n,k=1, is_sparse=False):
    if is_sparse:
        return sparse.eye_array(n, k=k)-sparse.eye_array(n)
    else:
        return np.eye(n, k=k)-np.eye(n)

def sparse_place_mat_on_diag(matrix: np.ndarray):
    mat_flat = matrix.ravel()
    return sparse.diags_array(mat_flat)

def get_matrix_gradient(matrix: np.ndarray, 
                        dx: np.ndarray, 
                        dy: np.ndarray):
    delx = matrix @ dx
    dely = dy @ matrix

    return delx, dely

def soft_shrinkage(matrix: np.ndarray,
                   tau: float):
    return np.sign(matrix)*np.max(np.abs(matrix)-tau,0)