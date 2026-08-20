# implements checking/evaluation functions

# packages ====================================================================
import numpy as np

# functions ===================================================================
def evaluate_progress_convex(f_k, f_k_1, x_k, x_k_1):
    """
    Convex sufficiently smooth objective implies zero gradient at optimal point 
    If the step size is small enough, this is effectively just approximate gradient
    Args:
        f_k (float): current function evaluation
        f_k_1 (float): previous function evaluation
        x_k (numpy array): current point
        x_k_1 (numpy array): previous point
    Returns:
        difference (float): how close the optimiser is to the optimal point
    """
    assert x_k.shape == x_k_1.shape, f"x_k dimension {x_k.shape} != x_k_1 dimension {x_k_1.shape}"

    function_value_difference = np.abs(f_k - f_k_1)
    point_difference = np.linalg.norm(
    	x_k - x_k_1, ord=2
    )
    
    difference = abs(function_value_difference / point_difference)
    return difference