from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim    
import numpy as np 
import matplotlib.pyplot as plt 
import pickle 
from copy import deepcopy 
from datetime import datetime
from tensorboardX import SummaryWriter
import io 
from PIL import Image 

def get_n_section_color(idx, N):
    """
    Returns a color from the rainbow colormap, split into N sections.
    
    :param idx: The index for which the color is required. Should be less than N.
    :param N: The total number of sections to divide the colormap into.
    :return: A tuple representing the color at the index location in RGB format.
    """
    if not (0 <= idx < N):
        raise ValueError("idx must be in the range [0, N-1]")
    
    # Use the 'hsv' colormap, which is a rainbow colormap
    cmap = plt.get_cmap('hsv')
    
    # Compute the color at the specific index
    color = cmap(float(idx) / N)
    
    return color


def execute_single_trial(origin, target, thresh): 
    
    # We have the origin and the starting point, now 
    # we just need to run the simulation 
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=target)
    
    # Then conduct the simulation 
    total_time = 5.0 
    dt = 0.1 
    
    # Calculate the crossing value 
    y_crossing = origin[1] - thresh 
    has_crossed_threshold = False
    
    # Also create data variables to store final return values 
    state_at_crossing = None 
    total_data_log = None 
    
    # Run the simulation 
    while (clpqr.time < total_time): 
        clpqr.step(dt) 
        
        # Check if the current state has crossed the threshold and hasn't before 
        current_state = clpqr.get_current_state()["state"]
        
        if current_state[1] < y_crossing and not has_crossed_threshold: 
            has_crossed_threshold = True
            
            # Save correlation states. 
            state_at_crossing = current_state
            
    # Once simulation is done, get the entire state log 
    total_data_log = clpqr.get_data_log()
    
    return state_at_crossing, total_data_log 
    
    
def calculate_bounce_jacobian(origin, target, thresh, dQ): 
    
    # We are going to calcualte the jacobian of the bounce 
    # by making small changes to the target, and seeing how they change 
    # the bounce conditions 
    
    # First, we need to get the bounce conditions at the current target 
    state_at_crossing, total_data_log = execute_single_trial(origin, target, thresh)

    # Debug printing 
    # print("Calculating bounce jacobian") 
    # print(f"STATE AT CROSSING: {state_at_crossing}")
    
    # Explicitly, these values are the x, position and theta
    x_at_crossing_0 = state_at_crossing[0]
    theta_at_crossing_0 = state_at_crossing[2]
    
    # Then we are going to perturb the target to change in each direction 
    # of the setpoint 
    jacobian_targets = [deepcopy(target), deepcopy(target)]
    
    # A small change in x position of setpoint 
    jacobian_targets[0][0] += dQ
    
    # A small change in y posiiton of setpoint 
    jacobian_targets[1][1] += dQ
    
    # Now we need to run the simulation for each of these targets   
    jacobian_states = []
    
    # Run the simulation for each target
    print(f"Running simulations for jacobian calculation, dq = {dQ}")
    print(f"Jacobian targets: {jacobian_targets}")
    for jacobian_target in jacobian_targets:
        
        print(f"Running simulation for target: {jacobian_target}")
        
        jacobian_state, _ = execute_single_trial(origin, jacobian_target, thresh)
        
        print(f"Results in state: {jacobian_state}, from oriignal state: {state_at_crossing}")
        
        jacobian_states.append(jacobian_state)
        
    # Now we need to calculate the jacobian, which will be a 2x2 matrix 
    # The first row is the change in x position at bounce due to change in x/y target
    # The second row is the change in theta at boucne due to change in x/y target 
    jacobian_matrix = np.zeros((2, 2)) 
    
    # Fill values 
    jacobian_matrix[0, 0] = (jacobian_states[0][0] - x_at_crossing_0) / dQ
    jacobian_matrix[0, 1] = (jacobian_states[1][0] - x_at_crossing_0) / dQ
    jacobian_matrix[1, 0] = (jacobian_states[0][2] - theta_at_crossing_0) / dQ
    jacobian_matrix[1, 1] = (jacobian_states[1][2] - theta_at_crossing_0) / dQ
    
    # Then return jacobian matrix 
    return jacobian_matrix 

def calculate_current_loss(goal_position, goal_pitch, state_at_crossing): 
    position_loss = goal_position - state_at_crossing[0]
    pitch_loss = goal_pitch - state_at_crossing[2]
    total_loss = np.sqrt(position_loss**2 + pitch_loss**2)
    
    return total_loss, position_loss, pitch_loss


def main(): 
    
    # Create a datetime string
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Append the datetime string to the folder name
    log_dir = f'runs/optimization_experiment_{current_time}'

    # Initialize the SummaryWriter with the new log directory
    writer = SummaryWriter(log_dir)
    
    # Define the origin of the simulation 
    origin = np.array([-2, 1, 6, 0, 0, 0]) 
    
    # Define the starting point of the optimization 
    initial_point = np.array([0, -1, 0, 0, 0, 0], dtype=np.float64)
    current_point = deepcopy(initial_point) 
    
    # Also define goal conditions: 
    goal_position = 0 # x = 0 
    goal_pitch = np.deg2rad(25) # 25 degrees 
    
    # Then we need to begin optimization 
    # Start by picking a step size for both the jacobian 
    # and the optimizer
    jacobian_step_size = 0.0001
    optimizer_step_size = 0.0001
    
    # Then we need to define the threshold for the y position 
    y_thresh = 1.5
    
    # Then we need to define the max# number of iterations
    num_iterations = 0 
    max_iterations = 1000
    
    # Then we need to define the convergence threshold
    current_loss = np.inf
    losses = [] 
    convergence_thresh = 0.01
    
    # Also define threshold for jacobian being noninvertible 
    jacobian_invertible_thresh = 1e-6
    
    # Now do the optimization 
    while(num_iterations <= max_iterations or current_loss > convergence_thresh): 
        # Incremenet number of iterations 
        num_iterations += 1
        
        # First calculate the current loss for this iteration 
        state_at_crossing, total_data_log = execute_single_trial(origin, current_point, y_thresh)
        current_loss, position_loss, pitch_loss = calculate_current_loss(goal_position, goal_pitch, state_at_crossing) 
        error_vector = np.array([position_loss, pitch_loss]).T 
        
        # Also add current_loss to losses 
        losses.append(current_loss)
        
        # Otherwise, calculate jacobian and update 
        jacobian = calculate_bounce_jacobian(origin, current_point, y_thresh, jacobian_step_size)
        
        # Check to see if the jacobian is invertible 
        # PASS for now 
        
        jacobian_inverse = np.linalg.inv(jacobian)
        
        # Then update the current point 
        # Create a point vector, which is a 2D representation of the current point 
        # The first row is the x position of the waypoint, the second row is y position of the waypoint
        current_point_vector = np.array([current_point[0], current_point[1]]).T
        
        # Then create a new vector which we will use to replace current_point 
        current_point_vector = current_point_vector + optimizer_step_size * (jacobian_inverse @ error_vector)
        
        # Then we use this vector to update the current waypoint
        current_point[0:2] = current_point_vector.T
        
        # Log data to the writer 
        writer.add_scalar('Total Loss', current_loss, num_iterations) 
        writer.add_scalar('Position Loss', position_loss, num_iterations)
        writer.add_scalar('Pitch Loss', pitch_loss, num_iterations)
        
        # Also add 2 views of the current point to see where it goes in the map 
        writer.add_scalar('Current Point X', current_point[0], num_iterations)
        writer.add_scalar('Current Point Y', current_point[1], num_iterations)
        
        # Create a scatter plot for X and Y 
        plt.figure() 
        plt.scatter(current_point[0], current_point[1], color='r')
        plt.grid() 
        plt.xlabel("X Position of Waypoint") 
        plt.ylabel("Y Position of Waypoint") 
        plt.xlim([-3, 4]) 
        plt.ylim([-3.5, 1.5])
        
        # Convert to image 
        buf = io.BytesIO() 
        plt.savefig(buf, format='png') 
        buf.seek(0) 
        image = Image.open(buf)
        image = np.array(image) 
        
        # Remove alpha channel 
        if image.shape[-1] == 4: 
            image = image[:, :, :3]
            
        # Add batch dimension 
        # image = np.expand_dims(image, axis=0) 
        image = np.transpose(image, (2, 0, 1))
            
        # Print image shape before adding to write r
        print(f"Image shape: {image.shape}")
            
        # Add image to writeer
        writer.add_image('Current Waypoint Position', image, num_iterations)
        
        writer.flush() # Push data to the dashboard 
        
        # Print out current statistics 
        print(f"TRIAL {num_iterations}, LOSS: {current_loss}, POSITION LOSS: {position_loss}, PITCH LOSS: {pitch_loss}, CURRENT POINT: {current_point}, JACOBIAN_DET:{np.linalg.det(jacobian)}")

        
if __name__ == "__main__": 
    main() 
        