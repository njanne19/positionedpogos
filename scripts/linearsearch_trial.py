from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim
import numpy as np
import matplotlib.pyplot as plt 
import pickle 
from scipy.interpolate import griddata

def main(): 
    
    # Define starting points and grids to search over
    origin = np.array([-2, 1, 6, 0, 0, 0])
    
    # Define grid of setpoints to simulate ove
    x_min = -1.5 # Min of grid x
    x_max = 3 # Max of grid x
    num_x = 1000 # Dimension 1 of grid 
    
    # Only use a single y value for the grid (the point below the virtual floor)
    y_min = -1
    
    # Then form mesh 
    xx = np.linspace(x_min, x_max, num_x)
    yy = np.ones(num_x) * y_min
    
    # Then turn every point in the mesh into a 6-dof setpoint 
    targets = np.array([xx.flatten(), yy.flatten(), np.zeros(num_x), np.zeros(num_x), np.zeros(num_x), np.zeros(num_x)]).T
    
    # Create a figure to plot all trajectories 
    trajectory_fig, trajectory_ax = plt.subplots()
    trajectory_ax.set_title("Grid Search Trajectories for $\Delta t = 5$",) 
    trajectory_ax.set_xlabel("x (m)") 
    trajectory_ax.set_ylabel("y (m)") 
    trajectory_ax.grid() 
    trajectory_ax.axis("equal")
    
    
    # Display setpoints on graph and initial conditions 
    trajectory_ax.scatter(targets[:, 0], targets[:, 1], label="Setpoints", c=[[1, 0, 0]])
    
    # Display origin on graph as a green dot 
    trajectory_ax.scatter(origin[0], origin[1], label="Origin", c = [[0, 1, 0]], zorder=10, s=100)
    
    # Create pickle file to save simulation data to 
    pickle_file = open(f"./out/linearsearch_{num_x}_{int(origin[2])}.pkl", "wb") 
    fit_file = open(f"./out/linearsearch_{num_x}_{int(origin[2])}_fit.pkl", "wb")
    
    # Create data struct for this pickle file 
    runs = {}
    
    # Create figures for correlation 
    correlation_fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    correlation_axes = correlation_fig.subplots(2, 2)
    
    # Create figures for the 2D contour. 
    contour_fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    contour_axis = contour_fig.subplots()
    
    ###### BEGIN PLOT FORMATTING ######
    
    # Correlation Axis setup 
    correlation_axes[0, 0].set_title("X Setpoint Offset vs. Pitch at Crossing")
    correlation_axes[0, 0].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[0, 0].set_ylabel("Pitch at Crossing (deg)")
    correlation_axes[0, 0].grid()
    
    correlation_axes[0, 1].set_title("X Setpoint Offset vs. X Position at Crossing")
    correlation_axes[0, 1].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[0, 1].set_ylabel("X Position at Crossing (m)")
    correlation_axes[0, 1].grid()
    
    correlation_axes[1, 0].set_title("X Setpoint Offset vs. X Velocity at Crossing")
    correlation_axes[1, 0].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[1, 0].set_ylabel("(Absolute) X Velocity at Crossing (m/s)")
    correlation_axes[1, 0].grid()
    
    correlation_axes[1, 1].set_title("X Setpoint Offset vs. Y Velocity at Crossing")
    correlation_axes[1, 1].set_xlabel("X Setpoint Offset (m)")
    correlation_axes[1, 1].set_ylabel("(Absolute) Y Velocity at Crossing (m/s)")
    correlation_axes[1, 1].grid()
    
    # Contour axis setup
    contour_axis.set_title("Ideal X Setpoint Offset for Requested Pitch/X Position at Crossing")
    contour_axis.set_xlabel("X Position at Crossing (m)")
    contour_axis.set_ylabel("Pitch at Crossing (deg)")
    
    ###### END PLOT FORMATTING ######
    
    # Create correlation data 
    correlation_data = np.zeros((num_x, 8))
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
                
                # Save all 6 state variables at this crossing
                correlation_data[i, 2:] = current_state
                
                # Make velocities absolute
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
        
        # Plot the correlation data to the graphs as a scatter 
        # X offset vs pitch at crossing
        correlation_axes[0, 0].plot(correlation_data[i, 0], correlation_data[i, 4], 'kx')
        # X offset vs X position at crossing
        correlation_axes[0, 1].plot(correlation_data[i, 0], correlation_data[i, 2], 'kx')

        # X offset vs X veloicity at crossing
        correlation_axes[1, 0].plot(correlation_data[i, 0], correlation_data[i, 5], 'kx')
        # X offset vs Y velocity at crossing
        correlation_axes[1, 1].plot(correlation_data[i, 0], correlation_data[i, 6], 'kx')
    
    
    # After the correlation data has been corrected, fit the correlation and 
    # plot fit + display parameters on the original plots 
    # X offset vs pitch at crossing
    x_pitch_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 4], 1)
    x_pitch_fit_fn = np.poly1d(x_pitch_fit)
    correlation_axes[0, 0].plot(correlation_data[:, 0], x_pitch_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[0, 0].text(0.05, 0.95, f"Fit: {x_pitch_fit_fn}", transform=correlation_axes[0, 0].transAxes, fontsize=10, verticalalignment='top')
    
    # X offset vs X position at crossing
    x_xp_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 2], 1)
    x_xp_fit_fn = np.poly1d(x_xp_fit)
    correlation_axes[0, 1].plot(correlation_data[:, 0], x_xp_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[0, 1].text(0.05, 0.95, f"Fit: {x_xp_fit_fn}", transform=correlation_axes[0, 1].transAxes, fontsize=10, verticalalignment='top')
    
    # X offset vs X velocity at crossing
    x_xvel_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 5], 1)
    x_xvel_fit_fn = np.poly1d(x_xvel_fit)
    correlation_axes[1, 0].plot(correlation_data[:, 0], x_xvel_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[1, 0].text(0.05, 0.95, f"Fit: {x_xvel_fit_fn}", transform=correlation_axes[1, 0].transAxes, fontsize=10, verticalalignment='top')
    
    # X offset vs Y velocity at crossing
    x_yvel_fit = np.polyfit(correlation_data[:, 0], correlation_data[:, 6], 1)
    x_yvel_fit_fn = np.poly1d(x_yvel_fit)
    correlation_axes[1, 1].plot(correlation_data[:, 0], x_yvel_fit_fn(correlation_data[:, 0]), '--', c='r')
    correlation_axes[1, 1].text(0.05, 0.95, f"Fit: {x_yvel_fit_fn}", transform=correlation_axes[1, 1].transAxes, fontsize=10, verticalalignment='top')
    
    
    #### 2D CONTOURS ####
    # 0. Scatter the original (x position at crossing, pitch) data points on top of the filled contour plot.
    contour_axis.scatter(correlation_data[:, 2], correlation_data[:, 4], c='k', s=30, zorder=10, label="Trial Data Points")
    
    # 1. Define a regular grid that covers your domain (x position at crossing, pitch).
    grid_x_at_crossing, grid_pitch_at_crossing = np.mgrid[correlation_data[:, 2].min():correlation_data[:, 2].max():100j,
                            correlation_data[:, 4].min():correlation_data[:, 4].max():100j]

    # For each of your data columns, do the following:

    # 2. Interpolate the data.
    grid_ideal_x_setpoint = griddata(correlation_data[:, [2, 4]], correlation_data[:, 0], (grid_x_at_crossing, grid_pitch_at_crossing), method='linear')

    # Save the fit data for future use
    fit_data = {
        "grid_x_at_crossing": grid_x_at_crossing,
        "grid_pitch_at_crossing": grid_pitch_at_crossing,
        "grid_ideal_x_setpoint": grid_ideal_x_setpoint
    }

    # 3. Visualize the interpolated data using contourf.
    contour_levels = 50  # Number of contour levels - you can adjust this value or provide specific levels
    plotted_contours = contour_axis.contourf(grid_x_at_crossing, grid_pitch_at_crossing, grid_ideal_x_setpoint, contour_levels, cmap='viridis')
    # Add color bar
    contour_fig.colorbar(plotted_contours, ax=contour_axis)
    
    # Finally, set axes of contour plot so that all points are shown 
    contour_axis.set_xlim(correlation_data[:, 2].min(), correlation_data[:, 2].max())
    contour_axis.set_ylim(correlation_data[:, 4].min(), correlation_data[:, 4].max())
    
    
    # Drone visualization
    # Plot y_thresh on the flightpath plot using the xlim set by matplotlib
    trajectory_ax.plot(trajectory_ax.get_xlim(), [origin[1] - y_thresh, origin[1] - y_thresh], 'r--', label="Y Threshold", linewidth=3, zorder=1)
    # plot gap between origin and threshold line 
    trajectory_ax.plot([origin[0], origin[0]], [origin[1], origin[1] - y_thresh], 'r--', linewidth=3, zorder=1)
    
    # Write data struct to pickle file
    pickle.dump(runs, pickle_file)
    pickle.dump(fit_data, fit_file)
    
    # Close pickle file 
    pickle_file.close()
    fit_file.close()
    
    # Display figure 
    trajectory_ax.legend() 
    plt.show() 
    
    
if __name__ == "__main__": 
    main() 