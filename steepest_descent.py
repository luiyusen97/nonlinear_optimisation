# implements steepest descent with backtracking line search

# packages ===============
import numpy as np
from tqdm import tqdm
import line_search
import approximate_derivatives
import evaluation_functions

# functions ===============
def steepest_descent(x_0, f_x, eps=1e-8, max_iter=1000000, verbose=False):
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
        p_k = -approximate_derivatives.approximate_gradient(f_x=f_x, x=x_k)

        alpha_k, backtracking_converged = line_search.line_search(
        	x_k=x_k, p_k=p_k, f_x=f_x, verbose=False
        )
        assert backtracking_converged, "Step length optimisation failed"
        proposed_step = alpha_k * p_k
        x_k += proposed_step
        steps_taken.append(x_k)
        function_evaluations.append(f_x(x_k))
        
        # difference_with_optimal_point = evaluation_functions.evaluate_progress_convex(
        # 	f_k=function_evaluations[-1], 
        # 	f_k_1=function_evaluations[-2], 
        # 	x_k=steps_taken[-1],
        # 	x_k_1=steps_taken[-2]
        # )
        difference_with_optimal_point = np.linalg.norm(approximate_derivatives.approximate_gradient(f_x=f_x, x=x_k), 2)
        if verbose:
            progress_bar.set_description(f"Improvement: {difference_with_optimal_point:.1g}")
        
        differences.append(difference_with_optimal_point)
        if np.abs(difference_with_optimal_point) < eps:
            if verbose:
                print("Optimiser complete")
            break
            
    return x_k, function_evaluations, steps_taken, differences