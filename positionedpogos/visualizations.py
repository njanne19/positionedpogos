import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation
import numpy as np 

def visualize_sim(closed_loop_system): 

    # Get data log from sim
    data_log = closed_loop_system.get_data_log()
    time = data_log["time"]
    state = data_log["state"]
    setpoint_vector = data_log["setpoint"]

    fig, ax = plt.subplots(2, 3)
    
    # Plot the 6-states of the system over time: 
    ax[0, 0].plot(
        time, 
        state[:, 0], 
        label="x"
    )
    ax[0, 1].plot(
        time, 
        state[:, 1], 
        label="y"
    )
    ax[0, 2].plot(
        time, 
        state[:, 2], 
        label="theta"
    )
    ax[1, 0].plot(
        time, 
        state[:, 3], 
        label="x_dot"
    )
    ax[1, 1].plot(
        time, 
        state[:, 4], 
        label="y_dot"
    )
    ax[1, 2].plot(
        time, 
        state[:, 5], 
        label="theta_dot"
    )
    
    for ax in ax.flatten(): 
        ax.legend()
        ax.grid() 
    
    
    # Now define quadcopter animation    
    fig2, ax2 = plt.subplots() 
    ax2.set_xlim(np.min(state[:, 0]) - 0.5, np.max(state[:, 0]) + 0.5)
    ax2.set_ylim(np.min(state[:, 1]) - 0.5, np.max(state[:, 1]) + 0.5)
        
    bar_length = 0.25
    quadcopter_bar, = ax2.plot([], [], 'b-', lw=2)
    
    # Set axis equal
    ax2.set_aspect('equal') 
    
    def quadcopter_init(): 
        quadcopter_bar.set_data([], [])
        return quadcopter_bar,
    
    def quadcopter_update(frame): 
        x = state[frame, 0]
        y = state[frame, 1]
        theta = state[frame, 2]
        
        # Compute the coordinates of the bar endpoints
        x_end = x + bar_length * np.cos(theta)
        y_end = y + bar_length * np.sin(theta)
        
        quadcopter_bar.set_data([x, x_end], [y, y_end])
        return quadcopter_bar,
    
    ani = FuncAnimation(
        fig2, 
        quadcopter_update, 
        frames=len(time), 
        init_func=quadcopter_init,
        blit=True, 
        interval = 1000*(time[1] - time[0])
    )
        
        
    if is_running_in_notebook(): 
        from IPython.display import HTML, display 
        display(fig) 
        display(HTML(ani.to_jshtml()))
        plt.close(fig2) 
    else: 
        plt.show()
    

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
