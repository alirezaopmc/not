"""
Mathematical functions and their derivatives for optimization examples
"""
import numpy as np
from typing import Tuple, Callable


def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
    """
    Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    
    This is a classic test function for optimization algorithms.
    Global minimum at (a, a²) with value 0.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector [x, y]
    a : float
        Parameter (default: 1.0)
    b : float  
        Parameter (default: 100.0)
        
    Returns:
    --------
    float
        Function value
    """
    return (a - x[0])**2 + b * (x[1] - x[0]**2)**2


def rosenbrock_gradient(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Gradient of Rosenbrock function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector [x, y]
    a : float
        Parameter (default: 1.0)
    b : float
        Parameter (default: 100.0)
        
    Returns:
    --------
    np.ndarray
        Gradient vector [df/dx, df/dy]
    """
    df_dx = -2 * (a - x[0]) - 4 * b * x[0] * (x[1] - x[0]**2)
    df_dy = 2 * b * (x[1] - x[0]**2)
    return np.array([df_dx, df_dy])


def rosenbrock_hessian(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
    """
    Hessian matrix of Rosenbrock function.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector [x, y]
    a : float
        Parameter (default: 1.0)
    b : float
        Parameter (default: 100.0)
        
    Returns:
    --------
    np.ndarray
        2x2 Hessian matrix
    """
    d2f_dx2 = 2 - 4 * b * (x[1] - x[0]**2) + 8 * b * x[0]**2
    d2f_dxdy = -4 * b * x[0]
    d2f_dy2 = 2 * b
    
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])


def quadratic_2d(x: np.ndarray, A: np.ndarray, b: np.ndarray, c: float = 0.0) -> float:
    """
    Quadratic function: f(x) = 0.5 * x^T A x + b^T x + c
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    A : np.ndarray
        Symmetric matrix (2x2)
    b : np.ndarray
        Linear term vector
    c : float
        Constant term
        
    Returns:
    --------
    float
        Function value
    """
    return 0.5 * x.T @ A @ x + b.T @ x + c


def quadratic_2d_gradient(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Gradient of quadratic function: ∇f(x) = A x + b
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    A : np.ndarray
        Symmetric matrix (2x2)
    b : np.ndarray
        Linear term vector
        
    Returns:
    --------
    np.ndarray
        Gradient vector
    """
    return A @ x + b


def beale_function(x: np.ndarray) -> float:
    """
    Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    
    Global minimum at (3, 0.5) with value 0.
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector [x, y]
        
    Returns:
    --------
    float
        Function value
    """
    term1 = (1.5 - x[0] + x[0] * x[1])**2
    term2 = (2.25 - x[0] + x[0] * x[1]**2)**2
    term3 = (2.625 - x[0] + x[0] * x[1]**3)**2
    return term1 + term2 + term3


def finite_difference_gradient(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute gradient using finite differences (forward difference).
    
    Parameters:
    -----------
    f : callable
        Function to differentiate
    x : np.ndarray
        Point at which to compute gradient
    h : float
        Step size for finite differences
        
    Returns:
    --------
    np.ndarray
        Approximate gradient
    """
    n = len(x)
    grad = np.zeros(n)
    
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (f(x_plus) - f(x)) / h
        
    return grad


def finite_difference_hessian(f: Callable, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """
    Compute Hessian using finite differences (central difference).
    
    Parameters:
    -----------
    f : callable
        Function to differentiate
    x : np.ndarray
        Point at which to compute Hessian
    h : float
        Step size for finite differences
        
    Returns:
    --------
    np.ndarray
        Approximate Hessian matrix
    """
    n = len(x)
    H = np.zeros((n, n))
    
    # Diagonal elements
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        H[i, i] = (f(x_plus) - 2*f(x) + f(x_minus)) / h**2
    
    # Off-diagonal elements
    for i in range(n):
        for j in range(i+1, n):
            x_pp = x.copy()
            x_pm = x.copy() 
            x_mp = x.copy()
            x_mm = x.copy()
            
            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h**2)
            H[j, i] = H[i, j]  # Symmetric
    
    return H