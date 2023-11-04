from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from scipy.interpolate import griddata


def main(): 
    
    # We are going to load in points from a fit, sample points
    # on that fit, and evaluate outcomes. 
    
    # Load in the fit
    fit_filename = "./out/linearsearch_100_6_fit.pkl"
    with open(fit_filename, "rb") as f: 
        fit = pickle.load(f)
        
    # Extract fit params 
    grid_x_at_crossing = fit["grid_x_at_crossing"]
    grid_pitch_at_crossing = fit["grid_pitch_at_crossing"]
    grid_ideal_x_setpoint = fit["grid_ideal_x_setpoint"]
    
    # Sample some random non-nan indices in this range which 
    # we can use for trials. 
    num_samples = 100
    non_nan_indices = np.argwhere(~np.isnan(grid_ideal_x_setpoint))
    # Sample indices in non-nan locations
    sample_indices = non_nan_indices[np.random.choice(non_nan_indices.shape[0], num_samples, replace=False)]
    
    # Then get the x position, theta, and pitch offset at these points
    x_at_crossing_sample = np.zeros(num_samples)
    pitch_at_crossing_sample = np.zeros(num_samples)
    ideal_x_setpoint_sample = np.zeros(num_samples)
    
    # Also define errors for these points
    x_error_sample = np.zeros(num_samples)
    pitch_error_sample = np.zeros(num_samples)
    
    for sample_num, idx in enumerate(sample_indices): 
        x_at_crossing_sample[sample_num] = grid_x_at_crossing[idx[0], idx[1]]
        pitch_at_crossing_sample[sample_num] = grid_pitch_at_crossing[idx[0], idx[1]]
        ideal_x_setpoint_sample[sample_num] = grid_ideal_x_setpoint[idx[0], idx[1]]
    
    # Plot the fit and randomly sample points on it 
    # to evaluate the fit
    fit_fig = plt.figure(figsize=(10, 10))
    fit_ax = fit_fig.add_subplot(111)
    
    # Plot the 2D contour data
    contour_levels = 50 
    plotted_contours = fit_ax.contourf(grid_x_at_crossing, grid_pitch_at_crossing, grid_ideal_x_setpoint, contour_levels)
    
    # Plot sample points exisiting on the contour 
    fit_ax.scatter(x_at_crossing_sample, pitch_at_crossing_sample, c='r', s=50, label='Evaluation Points')
    fit_ax.legend()
    
    # Add colorbar
    fit_fig.colorbar(plotted_contours, ax=fit_ax)
    
    # Format plot
    fit_ax.set_xlabel("X Position at Crossing (m)")
    fit_ax.set_ylabel("Pitch at Crossing (deg)")
    
    
    # Now that this plot is done, let's set up simulators for all these points
    # Define starting points
    origin = np.array([-2, 1, 6, 0, 0, 0])
    
    y_min = -1
    y_thresh = 1.5
    
    # Define target array
    xx = ideal_x_setpoint_sample + origin[0]
    yy = np.ones(num_samples) * y_min
    targets = np.array([xx.flatten(), yy.flatten(), np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)]).T

    
    for i in range(len(targets)): 
        
        # Print status message
        print(f"Running simulation for setpoint {i+1} of {len(targets)}")
    
        # Create ClosedLoopPlanarQuadrotor instance with origin and setpoint
        clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR, initial_state=origin, initial_setpoint=targets[i])
    
        # Define total time of simulation
        total_time = 5.0
        dt = 0.1 
        
        # Define threshold for crossing
        y_crossing = origin[1] - y_thresh
        has_crossed_threshold = False 
        
        # Run simulation 
        while(clpqr.time < total_time): 
            clpqr.step(dt) 
            
            # Check if the current state has crossed the threshold
            current_state = clpqr.get_current_state()["state"]
            
            if current_state[1] < y_crossing and not has_crossed_threshold:
                has_crossed_threshold = True
                
                # Save errors at this time. 
                x_error_sample[i] = np.abs(current_state[0] - x_at_crossing_sample[i])
                pitch_error_sample[i] = np.abs(np.rad2deg(current_state[2]) - pitch_at_crossing_sample[i])
                
                break
            
    
    # Then, generate histograms of errors for both, and combined
    error_fig = plt.figure(figsize=(10, 10))
    error_axes = error_fig.subplots(3, 1)
    
    # First start with histogram of just x_error_sample
    error_axes[0].hist(x_error_sample, bins=10)
    error_axes[0].set_xlabel("X Crossing Error (m)")
    error_axes[0].set_ylabel("Frequency")
    error_axes[0].grid()
    
    # Then histogram for just pitch_error_sample
    error_axes[1].hist(pitch_error_sample, bins=10)
    error_axes[1].set_xlabel("Pitch Crossing Error (deg)")
    error_axes[1].set_ylabel("Frequency")
    error_axes[1].grid()
    
    # Then calculate a combined error histogram 
    combined_error = x_error_sample + pitch_error_sample
    error_axes[2].hist(combined_error, bins=10)
    error_axes[2].set_xlabel("Combined Crossing Error")
    error_axes[2].set_ylabel("Frequency")
    error_axes[2].grid()
    
    # Set figure title 
    error_fig.suptitle("Histogram of Crossing Errors")
    
    plt.show()
    
if __name__ == "__main__": 
    main()