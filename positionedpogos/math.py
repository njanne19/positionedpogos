import numpy as np 
import math 
from typing import Callable, List, Tuple, Union 


def calculate_jacobian_forward_difference(
    f: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    
    """ Uses finite difference to calculate the jacobian matrix using forward difference.
    
    Args: 
        f: A function that takes in a numpy array and returns a numpy array. 
        q: The numpy array to calculate the jacobian at. 
        h: The step size for the finite difference.
    
    """
    
    f0 = f(q) # First get the baseline output variable 
    m = len(f0) # Then get the size of outputs
    n = len(q) # Then get the size of inputs
    
    jacobian = np.zeros((m, n)) # Initialize the jacobian matrix
    
    # Then loop through each input variable
    for i in range(n): 
        # Calculate the forward difference 
        q_forward = q.copy() 
        q_forward[i] += h 
        f_forward = f(q_forward)
        
        # Then calculate the forward difference 
        jacobian[:, i] = (f_forward - f0) / h
    
    return jacobian

def calculate_jacobian_backward_difference(
    f: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    
    """ Uses finite difference to calculate the jacobian matrix using backward difference.
    
    Args: 
        f: A function that takes in a numpy array and returns a numpy array. 
        q: The numpy array to calculate the jacobian at. 
        h: The step size for the finite difference.
    
    """
    
    f0 = f(q) # First get the baseline output variable 
    m = len(f0) # Then get the size of outputs
    n = len(q) # Then get the size of inputs
    
    jacobian = np.zeros((m, n)) # Initialize the jacobian matrix
    
    # Then loop through each input variable
    for i in range(n): 
        # Calculate the backward difference 
        q_backward = q.copy() 
        q_backward[i] -= h 
        f_backward = f(q_backward)
        
        # Then calculate the backward difference 
        jacobian[:, i] = (f0 - f_backward) / h
    
    return jacobian

def calculate_jacobian_symmetric_difference(
    f: Callable[[np.ndarray], np.ndarray],
    q: np.ndarray, h: float = 1e-5) -> np.ndarray:
    
    """ Uses finite difference to calculate the jacobian matrix using symmetric difference.
    
    Args: 
        f: A function that takes in a numpy array and returns a numpy array. 
        q: The numpy array to calculate the jacobian at. 
        h: The step size for the finite difference.
    
    """
    
    f0 = f(q) # First get the baseline output variable 
    m = len(f0) # Then get the size of outputs
    n = len(q) # Then get the size of inputs
    
    jacobian = np.zeros((m, n)) # Initialize the jacobian matrix
    
    # Then loop through each input variable
    for i in range(n): 
        # Then calculate the forward difference
        q_forward = q.copy() 
        q_forward[i] += h 
        f_forward = f(q_forward)
        
        # Then calculate the backward difference 
        q_backward = q.copy() 
        q_backward[i] -= h 
        f_backward = f(q_backward)
        
        # Then calculate the symmetric difference 
        jacobian[:, i] = (f_forward - f_backward) / (2 * h)
    
    return jacobian