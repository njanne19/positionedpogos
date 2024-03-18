import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np 
from scipy.spatial import ConvexHull

def visualize_single_trial(closed_loop_system, **kwargs): 
    
    # Get the data log from the sim
    data_log = closed_loop_system.get_data_log()
    time = data_log["time"]
    state = data_log["state"]
    setpoint_vector = data_log["setpoint"]
    thrust = data_log["input"]
    
    # Get virtual floor states 
    has_crossed_threshold = data_log["has_crossed_threshold"]
    state_at_crossing = data_log["state_at_crossing"]
    time_at_crossing = data_log["time_at_crossing"]
    y_crossing = data_log["y_crossing"]
    
    # Create a figure to visualize the simulation
    fig, ax = plt.subplots()
    
    # Plot the quadrotor's trajectory 
    ax.plot(state[:, 0], state[:, 1], label="Quadrotor Trajectory")
    
    # Turn on grid, make axes equal
    ax.grid()
    ax.set_aspect('equal')
    
    # Add title and labels
    if kwargs.get("trial_id") is not None: 
        ax.set_title(f"Quadrotor Trajectory (Trial {kwargs.get('trial_id')})")
    else: 
        ax.set_title("Quadrotor Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    
    # Plot the virtual floor
    ax.axhline(y_crossing, color='r', linestyle='--', label="Virtual Floor")
    
    # Plot the setpoint
    ax.scatter(setpoint_vector[:, 0], setpoint_vector[:, 1], label="Setpoint", c='g')
    
    # Draw a drone at the virtual floor with the pitch angle corresponding to the state at crossing
    if has_crossed_threshold:
        ax.scatter(state_at_crossing[0], state_at_crossing[1], label="State at Crossing", c='k')
        ax.quiver(
            state_at_crossing[0], 
            state_at_crossing[1], 
            0.5*np.cos(state_at_crossing[2]), 
            0.5*np.sin(state_at_crossing[2]), 
            scale=5, 
            label="BodyX @Cross",
            color='red'
        )
        # Draw body y axis as well
        ax.quiver(
            state_at_crossing[0], 
            state_at_crossing[1], 
            0.5*np.cos(state_at_crossing[2] + np.pi/2), 
            0.5*np.sin(state_at_crossing[2] + np.pi/2), 
            scale=5, 
            label="BodyY @Cross", 
            color='green'
        )
        
   
    # Plot the initial state
    ax.scatter(state[0, 0], state[0, 1], label="Initial State", c='b')
    ax.legend() 
    
    # Make a second plot that shows timeseries components of x/y/theta of the drone
    fig2, ax2 = plt.subplots(3, 1, figsize=(10, 8), tight_layout=True)
    ax2[0].plot(time, state[:, 0], label="X Position")
    ax2[0].plot(time, setpoint_vector[:, 0], label="X Setpoint", color='r', linestyle='--')
    ax2[0].set_title("X Position and Setpoint")
    ax2[0].set_xlabel("Time (s)")
    ax2[0].set_ylabel("X Position (m)")
    # Annotate where crossing point occurs 
    if has_crossed_threshold:
        ax2[0].axvline(time_at_crossing, color='k', label="Crossing Time")
        ax2[0].scatter(time_at_crossing, state_at_crossing[0], label="Crossing Point", c='k')
    ax2[0].legend()
    ax2[0].grid()
    
    # Y position
    ax2[1].plot(time, state[:, 1], label="Y Position")
    ax2[1].plot(time, setpoint_vector[:, 1], label="Y Setpoint", color='g', linestyle='--')
    ax2[1].set_title("Y Position and Setpoint")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Y Position (m)")
    # Annotate where crossing point occurs
    if has_crossed_threshold:
        ax2[1].axvline(time_at_crossing, color='k', label="Crossing Time")
        ax2[1].scatter(time_at_crossing, state_at_crossing[1], label="Crossing Point", c='k')
    ax2[1].legend()
    ax2[1].grid()
    
    # Theta position
    if kwargs.get("use_rad") is not None: 
        ax2[2].plot(time, state[:, 2], label="Theta Position")
        ax2[2].plot(time, setpoint_vector[:, 2], label="Theta Setpoint", color='b', linestyle='--')
        ax2[2].set_title("Theta Position and Setpoint")
        ax2[2].set_xlabel("Time (s)")
        ax2[2].set_ylabel("Theta Position (rad)")
    else:
        ax2[2].plot(time, np.degrees(state[:, 2]), label="Theta Position")
        ax2[2].plot(time, np.degrees(setpoint_vector[:, 2]), label="Theta Setpoint", color='b', linestyle='--')
        ax2[2].set_title("Theta Position and Setpoint")
        ax2[2].set_xlabel("Time (s)")
        ax2[2].set_ylabel("Theta Position (deg)")
    
    # Annotate where crossing point occurs
    if has_crossed_threshold:
        ax2[2].axvline(time_at_crossing, color='k', label="Crossing Time")
        if kwargs.get("use_rad") is not None: 
            ax2[2].scatter(time_at_crossing, state_at_crossing[2], label="Crossing Point", c='k')
        else:
            ax2[2].scatter(time_at_crossing, np.degrees(state_at_crossing[2]), label="Crossing Point", c='k')
    ax2[2].legend()
    ax2[2].grid()
    
    
def visualize_search_space_points(targets, **kwargs): 
    
    # Start by plotting all target points in the search space, use
    # equal axis and add a grid, draw a boundary curve around all of them.
    
    fig, ax = plt.subplots()
    ax.scatter(targets[:, 0], targets[:, 1], label="Waypoints", c='b', s=5)
    ax.set_title("Search Space Waypoints")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.grid()
    
    if kwargs.get("xlim") is not None:
        ax.set_xlim(kwargs.get("xlim"))
    else:
        ax.set_xlim([-1.5, 4.5])
    if kwargs.get("ylim") is not None:
        ax.set_ylim(kwargs.get("ylim"))
    else: 
        ax.set_ylim([-3.5, 2.5])
    
    ax.set_aspect('equal')
    
    # Draw a smoothed version of the convex hull around the points 
    hull = ConvexHull(targets)
    
    for simplex in hull.simplices:
        ax.plot(targets[simplex, 0], targets[simplex, 1], 'r-')
        
    # Also draw the centroid of the hull
    centroid = np.mean(targets[hull.vertices], axis=0)
    ax.scatter(centroid[0], centroid[1], label="Centroid", c='r')
    ax.legend()
    
    
    
    

def visualize_all_sim_states(closed_loop_system): 

    # Get data log from sim
    print(closed_loop_system) 
    data_log = closed_loop_system.get_data_log()
    time = data_log["time"]
    state = data_log["state"]
    setpoint_vector = data_log["setpoint"]
    thrust = data_log["input"]

    flight_overview = plt.figure(figsize=(14, 12), constrained_layout=True)
    flight_overview.suptitle("Flight Overview", fontsize=24, fontweight='heavy')
    
    # Generate two subfigures of flight_overview
    subfigs = flight_overview.subfigures(1, 2, wspace=0.07)
    
    # For six-dof figures
    six_dof_figure = subfigs[0]
    six_dof_axes = six_dof_figure.subplots(6, 1)
    # Also create a save figure that we can save data to 
    six_dof_save_figure = plt.figure(figsize=(14,10))
    six_dof_save_axes = six_dof_save_figure.subplots(3, 3)
    six_dof_save_figure.suptitle("Six-Dof Data")
    
    
    # Blank Plot the 6-states of the system over time: 
    six_dof_save_axes[0, 0].plot(
        time, 
        state[:, 0], 
        label="x"
    )
    six_dof_save_axes[0, 1].plot(
        time, 
        state[:, 1], 
        label="y"
    )
    six_dof_save_axes[0, 2].plot(
        time, 
        state[:, 2], 
        label="theta"
    )
    six_dof_save_axes[1, 0].plot(
        time, 
        state[:, 3], 
        label="x_dot"
    )
    six_dof_save_axes[1, 1].plot(
        time, 
        state[:, 4], 
        label="y_dot"
    )
    six_dof_save_axes[1, 2].plot(
        time, 
        state[:, 5], 
        label="theta_dot"
    )
    six_dof_save_axes[2, 0].plot(
        time, 
        thrust[:, 0], 
        label="u1"
    )
    six_dof_save_axes[2, 1].plot(
        time, 
        thrust[:, 1], 
        label="u2"
    )
    
    # Trace of the 6 states of the system over time. 
    six_dof_traces = []
    six_dof_labels = ["x", "y", "theta", "x_dot", "y_dot", "theta_dot"]
    for i, ax in enumerate(six_dof_axes.flatten()): 
        ax.grid() 
        ax.set_xlim(six_dof_save_axes.flatten()[i].get_xlim())
        ax.set_ylim(six_dof_save_axes.flatten()[i].get_ylim())
        six_dof_traces.append(ax.plot([], [], 'b-', lw=2, label=six_dof_labels[i])[0])
        ax.legend()
        ax.set_title(f"State: {six_dof_labels[i]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{six_dof_labels[i]}")
    
    # Figure split to visualize thrust inputs 
    flight_figure, thrust_figure = subfigs[1].subfigures(2, 1, hspace=0.07)
    
    # Now define quadcopter animation
    flight_axis = flight_figure.subplots() 
    flight_axis.set_xlim(np.min(state[:, 0]) - 0.5, np.max(state[:, 0]) + 0.5)
    flight_axis.set_ylim(np.min(state[:, 1]) - 0.5, np.max(state[:, 1]) + 0.5)
        
    bar_length = 0.25
    quadcopter_bar, = flight_axis.plot([], [], 'b-', lw=3, label="")
    u1_arrow, = flight_axis.plot([], [], 'r-', lw=2, label="") 
    u2_arrow, = flight_axis.plot([], [], 'r-', lw=2, label="")
    setpoint = flight_axis.scatter(0, 0, s=100, c=[[0, 1, 0]], label="Setpoint")

    
    # Set axis equal
    flight_axis.set_aspect('equal') 
    flight_axis.grid()
    flight_axis.legend() 
    
    # Setup thrust figure 
    thrust_axis = thrust_figure.subplots(1, 2)
    thrust_axis[0].set_xlim(six_dof_save_axes[2, 1].get_xlim())
    thrust_axis[0].set_ylim(six_dof_save_axes[2, 1].get_ylim())
    thrust_axis[1].set_xlim(six_dof_save_axes[2, 0].get_xlim())
    thrust_axis[1].set_ylim(six_dof_save_axes[2, 0].get_ylim())
    
    thrust_traces = []
    thrust_labels = ["u2", "u1"]
    for i, ax in enumerate(thrust_axis.flatten()):
        ax.grid()
        thrust_traces.append(ax.plot([], [], 'r-', lw=2, label=thrust_labels[i])[0])
        ax.legend()
        ax.set_xlabel("Time (s)") 
        ax.set_ylabel("Thrust (N)") 
        
        if i == 0: 
            ax.set_title("Left Propeller (Thrust 2)") 
        elif i == 1: 
            ax.set_title("RightPropeller (Thrust 1)")
    
    
    def quadcopter_init(): 
        quadcopter_bar.set_data([], [])
        u1_arrow.set_data([], [])
        u2_arrow.set_data([], [])
        return quadcopter_bar, u1_arrow, u2_arrow, setpoint,
    
    def quadcopter_update(frame): 
        x = state[frame, 0]
        y = state[frame, 1]
        theta = state[frame, 2]
        current_thrust = thrust[frame]
        
        # Update setpoint
        setpoint.set_offsets([setpoint_vector[frame, 0], setpoint_vector[frame, 1]])
        
        # Compute the coordinates of the bar endpoints
        x_start = x - (bar_length / 2.0) * np.cos(theta)
        y_start = y - (bar_length / 2.0) * np.sin(theta)
        x_end = x + (bar_length / 2.0) * np.cos(theta)
        y_end = y + (bar_length / 2.0) * np.sin(theta)
        
        # Starting points
        u1_arrow_x_start = x_end
        u1_arrow_y_start = y_end
        u2_arrow_x_start = x_start
        u2_arrow_y_start = y_start
        
        # Ending points
        u1_scale = 0.1 * current_thrust[0]/2.38 # Hardcoded hover value for now
        u2_scale = 0.1 * current_thrust[1]/2.38 
        u1_arrow_x_end = u1_arrow_x_start - u1_scale * np.sin(theta)
        u1_arrow_y_end = u1_arrow_y_start + u1_scale * np.cos(theta) 
        u2_arrow_x_end = u2_arrow_x_start - u2_scale * np.sin(theta) 
        u2_arrow_y_end = u2_arrow_y_start + u2_scale * np.cos(theta)
        
        quadcopter_bar.set_data([x_start, x_end], [y_start, y_end])
        u1_arrow.set_data([u1_arrow_x_start, u1_arrow_x_end], [u1_arrow_y_start, u1_arrow_y_end])
        u2_arrow.set_data([u2_arrow_x_start, u2_arrow_x_end], [u2_arrow_y_start, u2_arrow_y_end])
        return quadcopter_bar, u1_arrow, u2_arrow, setpoint,
    
    def six_dof_init(): 
        for i, trace in enumerate(six_dof_traces): 
            trace.set_data([], [])
            
        for i, trace in enumerate(thrust_traces):
            trace.set_data([], [])
            
        return tuple(six_dof_traces) + tuple(thrust_traces)
    
    def six_dof_update(frame): 
        
        current_time = time[:frame]
        x = state[:frame, 0]
        y = state[:frame, 1]
        theta = state[:frame, 2]
        x_dot = state[:frame, 3]
        y_dot = state[:frame, 4]
        theta_dot = state[:frame, 5]
        
        for i, trace in enumerate(six_dof_traces): 
            if i == 0: 
                trace.set_data(current_time, x)
            elif i == 1: 
                trace.set_data(current_time, y)
            elif i == 2: 
                trace.set_data(current_time, theta)
            elif i == 3: 
                trace.set_data(current_time, x_dot)
            elif i == 4: 
                trace.set_data(current_time, y_dot)
            elif i == 5: 
                trace.set_data(current_time, theta_dot)
                
        for i, trace in enumerate(thrust_traces): 
            if i == 0: 
                trace.set_data(current_time, thrust[:frame, 1])
            elif i == 1: 
                trace.set_data(current_time, thrust[:frame, 0])
        
        return tuple(six_dof_traces) + tuple(thrust_traces)
    
    
    def flight_overview_init(): 
        return quadcopter_init() + six_dof_init()
    
    def flight_overview_update(frame):
        return quadcopter_update(frame) + six_dof_update(frame)
    
    ani = FuncAnimation(
        flight_overview, 
        flight_overview_update, 
        frames=len(time), 
        init_func=flight_overview_init,
        blit=True, 
        interval = 1000*(time[1] - time[0])
    )
        
        
    if is_running_in_notebook(): 
        from IPython.display import HTML, display 
        print("Running in notebook")
        display(flight_overview) 
        display(HTML(ani.to_jshtml()))
        plt.close(six_dof_save_figure) 
    else: 
        
        # hide the save figure from being shown (just want png of it)
        plt.close(six_dof_save_figure)
        plt.show(block=True)
    

def is_running_in_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter
