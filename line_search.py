# this file implements wolfe condition line search

# packages ====================================================================
import numpy as np
from tqdm import tqdm
import approximate_derivatives

# functions ===================================================================
def check_armijo(x_k, p_k, f_x, nabla_fx_k, alpha_k, c_1):
    """
    Implements check for Armijo condition (or the first Wolfe condition), given
    the step length and step direction
    Args:
        x_k (numpy vector): current point
        p_k (numpy vector): descent direction, where p^T_k \nabla f(x_k) < 0
        f_x (function): objective function, mapping from R_n to R
        nabla_fx_k (numpy vector): gradient at current point
        alpha_k (float): step length
        c_1: Wolfe condition parameter
    Returns:
        fulfilled (bool): True if Armijo condition is fulfilled
    """
    assert x_k.shape[0] == p_k.shape[0], f"x_k is in R_{x_k.shape[0]} while p_k is of R_{p_k.shape[0]}"
    assert nabla_fx_k.shape[0] == x_k.shape[0], f"x_k is in R_{x_k.shape[0]} while nabla_fx_k if in R_{nabla_fx_k.shape[0]}"

    left_hand_side = f_x(x_k + alpha_k * p_k)
    right_hand_side = f_x(x_k) + c_1 * alpha_k * np.dot(p_k, nabla_fx_k)

    fulfilled = True if left_hand_side <= right_hand_side else False

    return fulfilled

def check_wolfe(x_k, p_k, nabla_fx_k, nabla_fx_k_alpha_k_p_k, c_2):
    """
    Implements check for Wolfe condition (or the second Wolfe condition), given
    the step length and step direction
    Args:
        x_k (numpy vector): current point
        p_k (numpy vector): descent direction, where p^T_k \nabla f(x_k) < 0
        nabla_fx_k (numpy vector): gradient at current point
        nabla_fx_k_alpha_k_p_k (numpy vector): gradient at next proposed point
        c_2: Wolfe condition parameter
    Returns:
        fulfilled (bool): True if Wolfe condition is fulfilled
    """
    assert x_k.shape[0] == p_k.shape[0], f"x_k is in R_{x_k.shape[0]} while p_k is of R_{p_k.shape[0]}"
    assert nabla_fx_k.shape[0] == x_k.shape[0], f"x_k is in R_{x_k.shape[0]} while nabla_fx_k if in R_{nabla_fx_k.shape[0]}"
    assert nabla_fx_k_alpha_k_p_k.shape[0] == x_k.shape[0], f"x_k is in R_{x_k.shape[0]} while nabla_fx_k_alpha_k_p_k if in R_{nabla_fx_k.shape[0]}"

    left_hand_side = -np.dot(p_k, nabla_fx_k_alpha_k_p_k)
    right_hand_side = -c_2 * np.dot(p_k, nabla_fx_k)

    fulfilled = True if left_hand_side <= right_hand_side else False

    return fulfilled

def line_search(x_k, p_k, f_x, alpha_0 = 1, c_1 = 1e-4, c_2 = 0.9, tau = 0.9, max_iter = 100000, verbose=False):
    """
    Implements backtracking line search for Newton and quasi-Newton methods
    Sufficient conditions for a step length such that the next step has enough curvature to cause
    a reduction in the objective function
    Args:
        x_k (numpy vector): current point
        p_k (numpy vector): descent direction, where p^T_k \nabla f(x_k) < 0
        f_x (function): objective function, mapping from R_n to R
        alpha_0 (float): initial step length guess
        c_1, c_2 (float): Wolfe condition parameters, s.t. 0 < c_1 < c_2 < 1
        tau (float): tau \in (0, 1), backtracking parameter, determines how fast the step size is reduced
        max_iter (int): safety feature, maximum iterations for backtracking to converge
        verbose (bool): whether to print diagnostic messages
    Returns:
        alpha_k (float): final step length
        converged (bool): False if backtracking failed to converge
    """
    assert x_k.shape[0] == p_k.shape[0], f"x_k is in R_{x_k.shape[0]} while p_k is of R_{p_k.shape[0]}"

    m = np.dot(p_k, approximate_derivatives.approximate_gradient(f_x=f_x, x=x_k))
    assert (tau < 1) and (tau > 0), f"Backtracking parameter {tau} must be in (0, 1)"

    alpha_k = alpha_0
    progress_bar = tqdm(range(max_iter)) if verbose else range(max_iter)
    for backtrack_step in progress_bar:

        if verbose:
            progress_bar.set_description("Backtracking with step length {alpha_k}")

        fulfilled_armijo = check_armijo(
            x_k=x_k, p_k=p_k, f_x=f_x,
            nabla_fx_k=approximate_derivatives.approximate_gradient(f_x=f_x, x=x_k),
            alpha_k=alpha_k, c_1=c_1
        )

        if fulfilled_armijo:
            if verbose:
                progress_bar.set_description(f"Backtracking succeeded at step {backtrack_step}")
            break
        else:
            alpha_k = tau * alpha_k

    if backtrack_step >= max_iter:
        if verbose:
            print(f"Maximum iterations of {max_iter} exceeded, returning last known step length")
        converged = False
    else:
        converged = True

    return alpha_k, converged