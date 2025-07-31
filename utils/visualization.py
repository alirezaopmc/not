"""
Visualization utilities for numerical optimization
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from typing import Callable, Tuple, Optional


def plot_function_2d(f: Callable, x_range: Tuple[float, float], 
                    y_range: Tuple[float, float], num_points: int = 100,
                    title: str = "Function Plot") -> None:
    """
    Plot a 2D function as a surface plot.
    
    Parameters:
    -----------
    f : callable
        Function to plot, should accept (x, y) and return scalar
    x_range : tuple
        (min_x, max_x) range for x-axis
    y_range : tuple  
        (min_y, max_y) range for y-axis
    num_points : int
        Number of points along each axis
    title : str
        Plot title
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = f(X[i, j], Y[i, j])
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y') 
    ax1.set_zlabel('f(x,y)')
    ax1.set_title(f'{title} - 3D Surface')
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20)
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title(f'{title} - Contour Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_gradient_field(f: Callable, grad_f: Callable, 
                       x_range: Tuple[float, float], y_range: Tuple[float, float],
                       num_points: int = 20, title: str = "Gradient Field") -> None:
    """
    Plot gradient field with contours.
    
    Parameters:
    -----------
    f : callable
        Function to plot contours for
    grad_f : callable
        Gradient function, should return (df/dx, df/dy)
    x_range : tuple
        (min_x, max_x) range for x-axis
    y_range : tuple
        (min_y, max_y) range for y-axis
    num_points : int
        Number of arrows along each axis
    title : str
        Plot title
    """
    # Dense grid for contours
    x_dense = np.linspace(x_range[0], x_range[1], 100)
    y_dense = np.linspace(y_range[0], y_range[1], 100)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    Z_dense = np.zeros_like(X_dense)
    
    for i in range(100):
        for j in range(100):
            Z_dense[i, j] = f(X_dense[i, j], Y_dense[i, j])
    
    # Sparse grid for gradient arrows
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = np.linspace(y_range[0], y_range[1], num_points)
    X, Y = np.meshgrid(x, y)
    
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    for i in range(num_points):
        for j in range(num_points):
            grad = grad_f(X[i, j], Y[i, j])
            U[i, j] = grad[0]
            V[i, j] = grad[1]
    
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    contour = plt.contour(X_dense, Y_dense, Z_dense, levels=20, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot gradient field (negative for descent direction)
    plt.quiver(X, Y, -U, -V, angles='xy', scale_units='xy', scale=1, 
               color='red', alpha=0.7, width=0.003)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'{title} (Red arrows show steepest descent direction)')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def plot_optimization_path(f: Callable, path: np.ndarray, 
                          x_range: Tuple[float, float], y_range: Tuple[float, float],
                          title: str = "Optimization Path") -> None:
    """
    Plot optimization path on function contours.
    
    Parameters:
    -----------
    f : callable
        Function to plot contours for
    path : np.ndarray
        Array of shape (n_iterations, 2) containing the optimization path
    x_range : tuple
        (min_x, max_x) range for x-axis
    y_range : tuple
        (min_y, max_y) range for y-axis
    title : str
        Plot title
    """
    # Create contour plot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    for i in range(100):
        for j in range(100):
            Z[i, j] = f(X[i, j], Y[i, j])
    
    plt.figure(figsize=(10, 8))
    
    # Plot contours
    contour = plt.contour(X, Y, Z, levels=20, alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)
    
    # Plot optimization path
    plt.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, 
             label='Optimization Path')
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, label='Start')
    plt.plot(path[-1, 0], path[-1, 1], 'bs', markersize=10, label='End')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()