from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
import numpy as np 

def execute_single_trial(origin, target, threshold, saturation=True): 
    """
        This function will execute a single trial of the simulation.
        :param origin: The initial state of the quadrotor
        :param target: The target state of the quadrotor
        :param threshold: The offset between origin y coordinate that defines the virtual floor
        :param saturation: Whether or not the controller uses saturation
        :return: The state at which the threshold was crossed
    
    """
    
    # We have the origin and the starting point, now
    # we just need to run the simulation 
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=target, saturation=saturation)
    
    # Then conduct the simulation 
    total_time = 5.0 
    dt = 0.01
    dt_mini = 0.001
    use_dt_mini = False
    
    # Calculate the crossing value 
    clpqr.data_log['y_crossing'] = origin[1] - threshold
    clpqr.data_log['has_crossed_threshold'] = False 
    
    # Also create data variables to store the final return values
    clpqr.data_log['state_at_crossing'] = None
    
    # Run the simulation 
    while (clpqr.time < total_time): 
        
        # When approaching the crossing point, use smaller time step to ensure accuracy around this point. 
        if use_dt_mini: 
            clpqr.step(dt_mini)
        else: 
            clpqr.step(dt) 
        
        # Check if the current state has crossed the threshold and hasn't before
        current_state = clpqr.get_current_state()["state"]
        
        if current_state[1] < clpqr.data_log['y_crossing'] and not clpqr.data_log['has_crossed_threshold']: 
            clpqr.data_log['has_crossed_threshold'] = True 
            clpqr.data_log['state_at_crossing'] = current_state
            clpqr.data_log['time_at_crossing'] = clpqr.time
            use_dt_mini = False
        elif np.abs(current_state[1] - clpqr.data_log['y_crossing']) < 0.1 and not clpqr.data_log['has_crossed_threshold']: 
            use_dt_mini = True
    
    # If the threshold is never crossed, state_at_crossing returns None 
    return clpqr


def execute_trajectory_trial(origin, target, threshold, saturation=True): 
    """
        This function will execute a single trial of the simulation.
        :param origin: The initial state of the quadrotor
        :param target: The target state of the quadrotor
        :param threshold: The offset between origin y coordinate that defines the virtual floor
        :param saturation: Whether or not the controller uses saturation
        :return: The state at which the threshold was crossed
    
    """
    
    # We have the origin and the starting point, now
    # we just need to run the simulation 
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=target, saturation=saturation)
    
    # Then conduct the simulation 
    total_time = 5.0 
    dt = 0.01
    dt_mini = 0.001
    use_dt_mini = False
    
    # Calculate the crossing value 
    clpqr.data_log['y_crossing'] = origin[1] - threshold
    clpqr.data_log['has_crossed_threshold'] = False 
    
    # Also create data variables to store the final return values
    clpqr.data_log['state_at_crossing'] = None
    
    # Run the simulation 
    while (clpqr.time < total_time): 
        
        # When approaching the crossing point, use smaller time step to ensure accuracy around this point. 
        if use_dt_mini: 
            clpqr.step(dt_mini)
        else: 
            clpqr.step(dt) 
        
        # Check if the current state has crossed the threshold and hasn't before
        current_state = clpqr.get_current_state()["state"]
        
        if current_state[1] < clpqr.data_log['y_crossing'] and not clpqr.data_log['has_crossed_threshold']: 
            clpqr.data_log['has_crossed_threshold'] = True 
            clpqr.data_log['state_at_crossing'] = current_state
            clpqr.data_log['time_at_crossing'] = clpqr.time
            use_dt_mini = False
        elif np.abs(current_state[1] - clpqr.data_log['y_crossing']) < 0.1 and not clpqr.data_log['has_crossed_threshold']: 
            use_dt_mini = True
    
    # If the threshold is never crossed, state_at_crossing returns None 
    return clpqr



def generate_grid(x_limits, y_limits, n_points_x, n_points_y):
    """
    Generate a grid of X, Y points within given limits with a specified number
    of points along the X and Y axes.
    
    Parameters:
    - x_limits: Tuple of (min, max) for X-axis limits.
    - y_limits: Tuple of (min, max) for Y-axis limits.
    - n_points_x: Number of points along the X-axis.
    - n_points_y: Number of points along the Y-axis.
    
    Returns:
    - grid_points: Array of grid points (x, y).
    """
    # Generate linearly spaced points within the limits
    x_points = np.linspace(x_limits[0], x_limits[1], n_points_x)
    y_points = np.linspace(y_limits[0], y_limits[1], n_points_y)
    
    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_points, y_points)
    
    # Flatten the meshgrid arrays to get a list of points
    grid_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    
    return grid_points
