# implements steepest descent with backtracking line search

# packages ===============
import numpy as np
from tqdm import tqdm
import line_search
import approximate_derivative

# functions ===============
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
    
    function_value_difference = np.linalg.norm(
    	f_k - f_k, ord=2
    )
    point_difference = np.linalg.norm(
    	x_k - x_k_1, ord=2
    )
    
    difference = abs(function_value_difference / point_difference)
    return difference
	
def steepest_descent(x_0, f_x, eps, max_iter, verbose):
    """
    Implements steepest descent method for convex optimisation
    Args:
        x_0 (numpy array): R_n starting point
        f_x (function): maps from R_n to R
        eps (float): tolerance for stopping optimisation
        max_iter (int): maximum number of iterations, safety feature
        verbose (bool): True if intermediate prints needed
    Returns:
        x_k (numpy array): global minimum
        function_evaluations (list): tracks the progress down the objective function
        steps_taken (list): tracks the path taken in the domain
        differences (list): tracks the normalised improvements for each step
    """
    # main loop should be
    # get descent direction
    # get step length
    # take a step
    # evaluate objective function, save function value
    # compare current and previous function value
    # stop if normalised value is less than eps
    # also stop if max_iter is exceeded
    
    assert np.isreal(f_x(x_0)), f"Starting point {x_0} not in domain"
    
    # initialise caches
    function_evaluations = [f_x(x_0)]
    steps_taken = [x_0]
    differences = []
    
    x_k = x_0.copy()
    progress_bar = tqdm(range(max_iter)) if verbose else range(max_iter)
    for iter in progress_bar:
        p_k = approximate_derivative.approximate_gradient(f_x=f_x, x=x_k)
        alpha_k, backtracking_converged = line_search.line_search(
        	x_k=x_k, p_k=p_k, f_x=f_x
        )
        assert backtracking_converged, "Step length optimisation failed"
        proposed_step = alpha_k * p_k
        x_k += proposed_step
        steps_taken.append(x_k)
        function_evaluations.append(f_x(x_k))
        
        difference_with_optimal_point = evaluate_progress_convex(
        	f_k=function_evaluations[-1], 
        	f_k_1=function_evaluations[-2], 
        	x_k=steps_taken[-1],
        	x_k_1=steps_taken[-2]
        )
        if verbose:
            progress_bar.set_description(f"Improvement: {np.round(difference_with_optimal_point, 1)}")
        
        differences.append(difference_with_optimal_point)
        if np.abs(difference_with_optimal_point) < eps:
            if verbose:
                print("Optimiser complete")
            break
            
    return alpha_k, function_evaluations, steps_taken, differences