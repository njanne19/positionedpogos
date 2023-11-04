from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

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

def main(): 
    
    # Define starting points and grids to search over
    origin = np.array([-2, 1, 6, 0, 0, 0])
    
    # Define grid of setpoints to simulate over
    x_min = -1 # Min of grid x
    x_max = 3 # Max of grid x
    num_x = 100 # Dimension 1 of grid 
    
    y_min = -3 # Min of grid y
    y_max = -1 # Max of grid y 
    num_y = 10 # Dimension 2 of grid
    
    # Then form mesh 
    x = np.linspace(x_min, x_max, num_x)
    y = np.linspace(y_min, y_max, num_y)
    xx, yy = np.meshgrid(x, y)
    
    # Then turn every point in the mesh into a 6-dof setpoint 
    targets = np.array([xx.flatten(), yy.flatten(), np.zeros(num_x*num_y), np.zeros(num_x*num_y), np.zeros(num_x*num_y), np.zeros(num_x*num_y)]).T
    
    # print(targets)
    
    # Create a figure to plot all trajectories 
    trajectory_fig, trajectory_ax = plt.subplots()
    trajectory_ax.set_title("Grid Search Trajectories for $\Delta t = 5$",) 
    trajectory_ax.set_xlabel("x (m)") 
    trajectory_ax.set_ylabel("y (m)") 
    trajectory_ax.grid() 
    trajectory_ax.axis("equal")
    
    # Create a 3D plot to show trajectories across setpoint configurations
    trajectory_fig_3d = plt.figure(figsize=(12, 8), constrained_layout=True)
    trajectory_ax_3d = trajectory_fig_3d.add_subplot(121, projection='3d')
    trajectory_ax_3d.set_title("3D Trajectories for $\Delta t = 5$")
    trajectory_ax_3d_single = trajectory_fig_3d.add_subplot(122, projection='3d')
    trajectory_ax_3d_single.set_title("Subset of 3D Trajectories for $\Delta t = 5$")
    
    
    # Display setpoints on graph and initial conditions 
    trajectory_ax.scatter(targets[:, 0], targets[:, 1], label="Setpoints", c=[[1, 0, 0]])
    
    # Display origin on graph as a green dot 
    trajectory_ax.scatter(origin[0], origin[1], label="Origin", c = [[0, 1, 0]], zorder=10, s=100)
    
    # Create pickle file to save simulation data to 
    pickle_file = open(f"./out/gridsearch_{num_x}_{num_y}_{int(origin[2])}.pkl", "wb") 
    
    # Create data struct for this pickle file 
    runs = {}
    
    # Create figures for correlation 
    correlation_fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    correlation_axes = correlation_fig.subplots(2, 3)
    
    ###### BEGIN PLOT FORMATTING ######
    
    # Axis setup 
    # setpoint_offset_x, (theta at crossing, x velocity at crossing, y velocity at crossing)
    # setpoint_offset_y, (theta at crossing, x velocity at crossing, y velocity at crossing)
    correlation_axes[0, 0].set_title("X Setpoint Offset vs. Pitch at Crossing")
    correlation_axes[0, 0].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[0, 0].set_ylabel("Pitch at Crossing (deg)")
    correlation_axes[0, 0].grid()
    
    correlation_axes[0, 1].set_title("X Setpoint Offset vs. X Position at Crossing")
    correlation_axes[0, 1].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[0, 1].set_ylabel("X Velocity at Crossing (m/s)")
    correlation_axes[0, 1].grid()
    
    correlation_axes[0, 2].set_title("X Setpoint Offset vs. X Velocity at Crossing")
    correlation_axes[0, 2].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[0, 2].set_ylabel("Y Velocity at Crossing (m/s)")
    correlation_axes[0, 2].grid()
    
    correlation_axes[1, 0].set_title("Y Setpoint Offset vs. Pitch at Crossing")
    correlation_axes[1, 0].set_xlabel("Y Setpoint Offset (m)")
    correlation_axes[1, 0].set_ylabel("Pitch at Crossing (deg)")
    correlation_axes[1, 0].grid()
    
    correlation_axes[1, 1].set_title("Y Setpoint Offset vs. X Position at Crossing")
    correlation_axes[1, 1].set_xlabel("Y Setpoint Offset (m)")
    correlation_axes[1, 1].set_ylabel("X Velocity at Crossing (m/s)")
    correlation_axes[1, 1].grid()
    
    correlation_axes[1, 2].set_title("Y Setpoint Offset vs. Y Velocity at Crossing")
    correlation_axes[1, 2].set_xlabel("Y Setpoint Offset (m)")
    correlation_axes[1, 2].set_ylabel("Y Velocity at Crossing (m/s)")
    correlation_axes[1, 2].grid()
    
    
    
    ###### END PLOT FORMATTING ######
    
    # Create correlation data 
    correlation_data = np.zeros((num_x*num_y, 8))
    y_thresh = 1.5 # Threshold for y value to cross
    
    # Fill out the first two columns of correlation data (offset from origin) 
    correlation_data[:, 0] = targets[:, 0] - origin[0]
    correlation_data[:, 1] = targets[:, 1] - origin[1]
    
    # Then actually run the simulation 
    for i in range(len(targets)): 
        
        # Print status message
        print(f"Running simulation for setpoint {i+1} of {len(targets)}")
        
        # Create ClosedLoopPlanarQuadrotor instance with origin and setpoint 
        clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=targets[i])
        
        # Define total time of simulation 
        total_time = 5.0 
        dt = 0.1
        
        # Measure correlations at this y value 
        y_crossing = origin[1] - y_thresh
        has_crossed_threshold = False 
        
        # Run simulation 
        while (clpqr.time < total_time): 
            clpqr.step(dt) 
            
            # Check if the current state has crossed the threshold and hasn't before 
            current_state = clpqr.get_current_state()["state"]
            
            if current_state[1] < y_crossing and not has_crossed_threshold:
                has_crossed_threshold = True 
                
                # Save correlation states. 
                correlation_data[i, 2:] = current_state
                correlation_data[i, 5] = np.abs(correlation_data[i, 5])
                correlation_data[i, 6] = np.abs(correlation_data[i, 6])     
                
                # Change theta to degrees
                correlation_data[i, 4] = np.rad2deg(correlation_data[i, 4])
            
        # Collect simulation data log 
        data_log = clpqr.get_data_log()
            
        # Log data to data struct 
        runs[i] = {
            "target": targets[i],
            "log": data_log
        }
        
        # Plot the data to the graph as a semi-transparent dashed grey line 
        trajectory_ax.plot(data_log["state"][:, 0], data_log["state"][:, 1], 'k--', alpha=0.5)
        
        # Add trajectory to 3D plot as well
        # First index of the plot is the setpoint index 
        
        # print(f"Index is {i}, x is {targets[i][0]}, y is {targets[i][1]}, idx (mod num_y) is {i%num_y}, idx (mod num_x) is {i%num_x}")
        
        # Add trajectory to 3D plot
        trajectory_ax_3d.plot3D(np.ones(len(data_log["state"][:, 0]))*i, data_log["state"][:, 0], data_log["state"][:, 1], color=get_n_section_color(i%num_x, num_x))
        
        # Also add to subset if we are in the first numx trajectories 
        if i < num_x:
            trajectory_ax_3d_single.plot3D(np.ones(len(data_log["state"][:, 0]))*i, data_log["state"][:, 0], data_log["state"][:, 1], color=get_n_section_color(i%num_x, num_x))
        
        
        # Plot the correlation data to the graphs as a scatter 
        # X offset vs pitch at crossing
        correlation_axes[0, 0].plot(correlation_data[i, 0], correlation_data[i, 4], 'kx')
        # X offset vs x position at crossing
        correlation_axes[0, 1].plot(correlation_data[i, 0], correlation_data[i, 2], 'kx')
        # X offset vs x velocity at crossing
        correlation_axes[0, 2].plot(correlation_data[i, 0], correlation_data[i, 5], 'kx')

        # Y offset vs pitch at crossing
        correlation_axes[1, 0].plot(correlation_data[i, 1], correlation_data[i, 4], 'kx')
        # Y offset vs X velocity at crossing
        correlation_axes[1, 1].plot(correlation_data[i, 1], correlation_data[i, 2], 'kx')
        # Y offset vs Y velocity at crossing
        correlation_axes[1, 2].plot(correlation_data[i, 1], correlation_data[i, 6], 'kx')
        
    
    ### Model fitting 
    # Try and fit a 2D model to the data 
    # First collect data of interest 
    # Pitch at crossing and x position at crossing 
    X = correlation_data[:, [4, 2]]
    # X setpoint offset and Y setpoint offset
    Y = correlation_data[:, [0, 1]]
    
    # Generate train, test splits 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Then fit the model 
    model = RandomForestRegressor() 
    model.fit(X_train, Y_train) 
    
    # Predict on the test set
    Y_pred = model.predict(X_test)
    
    # Evaluate the model 
    mse = mean_squared_error(Y_test, Y_pred)
    
    # Show stats: 
    print(f"Random Forest Regressor Model MSE: {mse}")
    
    # Try on some new cases 
    num_evaluation_samples = 100
    
    # Generate evaluation samples by taking 100 random points in the bounds of the original
    # X and y points: 
    pitch_eval = np.random.uniform(np.min(correlation_data[:, 4]), np.max(correlation_data[:, 4]), num_evaluation_samples)
    x_setpoint_position_eval = np.random.uniform(np.min(correlation_data[:, 2]), np.max(correlation_data[:, 2]), num_evaluation_samples)
    
    # Get offsets assigned to these models 
    setpoint_offset_eval = model.predict(np.array([pitch_eval, x_setpoint_position_eval]).T)
    
    # print(setpoint_offset_eval)
    # print(setpoint_offset_eval.shape)
    
    # Then generate a new set of targets with these offsets 
    targets_eval = np.tile(origin, (num_evaluation_samples, 1)) + np.hstack((setpoint_offset_eval, np.zeros((num_evaluation_samples, 4))))
    
    # Create a new data array to store these values 
    evaluation_data = np.zeros((num_evaluation_samples, 8))
    evaluation_data[:, :2] = setpoint_offset_eval
    
    # Then run the simulation for these new targets 
    for i in range(len(targets_eval)):
        
        # Print status message
        print(f"Running simulation for evaluation setpoint {i+1} of {len(targets_eval)}")
    
        # Create ClosedLoopPlanarQuadrotor instance with origin and setpoint
        clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=targets_eval[i])
        
        # Define total time of simulation
        total_time = 5.0
        dt = 0.1
        
        # Measure correlations at this y value
        y_crossing = origin[1] - y_thresh
        has_crossed_threshold = False
        
        # Run simulation
        while (clpqr.time < total_time):
            clpqr.step(dt)
            
            # Check if the current state has crossed the threshold and hasn't before
            current_state = clpqr.get_current_state()["state"]
            
            if current_state[1] < y_crossing and not has_crossed_threshold:
                has_crossed_threshold = True
                
                # Save correlation states.
                evaluation_data[i, 2:] = current_state
                evaluation_data[i, 5] = np.abs(evaluation_data[i, 5])
                evaluation_data[i, 6] = np.abs(evaluation_data[i, 6])
                
                # Change theta to degrees
                evaluation_data[i, 4] = np.rad2deg(evaluation_data[i, 4])
                
                break
        
    
    # Then compare the resulting pitch at crossing and x position at crossing 
    # to the requested 
    # Pitch at crossing error 
    pitch_at_crossing_error = np.abs(evaluation_data[:, 4] - pitch_eval)
    # X position at crossing error
    x_position_at_crossing_error = np.abs(evaluation_data[:, 2] - x_setpoint_position_eval)
    
    # Plot errors on histograms 
    error_fig = plt.figure(figsize=(10, 10))
    error_axes = error_fig.subplots(3, 1)
    
    # First start with histogram of just x position error at crossing
    error_axes[0].hist(x_position_at_crossing_error, bins=10)
    error_axes[0].set_xlabel("X Crossing Error (m)")
    error_axes[0].set_ylabel("Frequency")
    error_axes[0].grid()
    
    # Then histogram for just pitch at crossing error
    error_axes[1].hist(pitch_at_crossing_error, bins=10)
    error_axes[1].set_xlabel("Pitch Crossing Error (deg)")
    error_axes[1].set_ylabel("Frequency")
    error_axes[1].grid()
    
    # Then calculate a combined error histogram
    combined_error = x_position_at_crossing_error + pitch_at_crossing_error
    error_axes[2].hist(combined_error, bins=10)
    error_axes[2].set_xlabel("Combined Crossing Error")
    error_axes[2].set_ylabel("Frequency")
    error_axes[2].grid()
    
    error_fig.suptitle("Evaluation of 2D Random Forest Model Fit")
    
    
    # After the correlation data has been corrected, fit the correlation and 
    # plot fit + display parameters on the original plots 
    # X offset vs pitch at crossing
    x_pitch_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 4], 1)
    x_pitch_fit_fn = np.poly1d(x_pitch_fit)
    correlation_axes[0, 0].plot(correlation_data[:, 0], x_pitch_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[0, 0].text(0.05, 0.95, f"Fit: {x_pitch_fit_fn}", transform=correlation_axes[0, 0].transAxes, fontsize=10, verticalalignment='top')
    
    # X offset vs X velocity at crossing
    x_xp_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 2], 1)
    x_xp_fit_fn = np.poly1d(x_xp_fit)
    correlation_axes[0, 1].plot(correlation_data[:, 0], x_xp_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[0, 1].text(0.05, 0.95, f"Fit: {x_xp_fit_fn}", transform=correlation_axes[0, 1].transAxes, fontsize=10, verticalalignment='top')
    
    # X offset vs X velocity at crossing
    x_xvel_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 5], 1)
    x_xvel_fit_fn = np.poly1d(x_xvel_fit)
    correlation_axes[0, 2].plot(correlation_data[:, 0], x_xvel_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[0, 2].text(0.05, 0.95, f"Fit: {x_xvel_fit_fn}", transform=correlation_axes[0, 2].transAxes, fontsize=10, verticalalignment='top')
    
    # Y offset vs pitch at crossing
    y_pitch_fit = np.polyfit(correlation_data[:, 1], correlation_data[:, 4], 1)
    y_pitch_fit_fn = np.poly1d(y_pitch_fit)
    correlation_axes[1, 0].plot(correlation_data[:, 1], y_pitch_fit_fn(correlation_data[:, 1]), '--', c='r')
    correlation_axes[1, 0].text(0.05, 0.95, f"Fit: {y_pitch_fit_fn}", transform=correlation_axes[1, 0].transAxes, fontsize=10, verticalalignment='top')
    
    # Y offset vs x position at crossing
    y_xp_fit = np.polyfit(correlation_data[:, 1], correlation_data[:, 2], 1)
    y_xp_fn = np.poly1d(y_xp_fit)
    correlation_axes[1, 1].plot(correlation_data[:, 1], y_xp_fn(correlation_data[:, 1]), '--', c='r')
    correlation_axes[1, 1].text(0.05, 0.95, f"Fit: {y_xp_fn}", transform=correlation_axes[1, 1].transAxes, fontsize=10, verticalalignment='top')
    
    # Y offset vs Y velocity at crossing
    y_yvel_fit = np.polyfit(correlation_data[:, 1], correlation_data[:, 6], 1)
    y_yvel_fit_fn = np.poly1d(y_yvel_fit)
    correlation_axes[1, 2].plot(correlation_data[:, 1], y_yvel_fit_fn(correlation_data[:, 1]), '--', c='r')
    correlation_axes[1, 2].text(0.05, 0.95, f"Fit: {y_yvel_fit_fn}", transform=correlation_axes[1, 2].transAxes, fontsize=10, verticalalignment='top')
   
    # Plot y_thresh on the flightpath plot using the xlim set by matplotlib
    trajectory_ax.plot(trajectory_ax.get_xlim(), [origin[1] - y_thresh, origin[1] - y_thresh], 'r--', label="Y Threshold", linewidth=3, zorder=1)
    # plot gap between origin and threshold line 
    trajectory_ax.plot([origin[0], origin[0]], [origin[1], origin[1] - y_thresh], 'r--', linewidth=3, zorder=1)
    
    # Write data struct to pickle file
    pickle.dump(runs, pickle_file)
    
    # Close pickle file 
    pickle_file.close()
    
    # Display figure 
    trajectory_ax.legend() 
    plt.show() 
    
    
if __name__ == "__main__": 
    main() 