# this file implements approximate derivative computations

# packages ====================================================================
import numpy as np

# functions ===================================================================
def approximate_gradient(f_x, x, h = 1e-4):
    """
    Computes the approximate gradient for a f_x that maps from R_n to R
    Args:
        f_x (function): maps R_n to R
        x (numpy array): R_n array
        h (float): tolerance for accuracy
    Returns:
        gradient (numpy array): R_n array
    """
    gradient = np.empty(shape=x.shape)
    for i in range(x.shape[0]):
        e_i = np.zeros(shape=x.shape)
        e_i[i] = 1
        backward_bound = f_x(x - h / 2 * e_i)
        forward_bound = f_x(x + h / 2 * e_i)
        gradient[i] = (forward_bound - backward_bound) / h

    return gradient