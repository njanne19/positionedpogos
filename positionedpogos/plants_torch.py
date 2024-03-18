import torch
import torch.nn as nn 

class PlanarQuadrotor(nn.Module): 
    def __init__(self, mode="two_thrusters"): 
        super(PlanarQuadrotor, self).__init__()

        # Quadrotor parameters, these are defaults from a paper I found online 
        self.m = torch.tensor(1.5) # Mass, kg
        self.I = torch.tensor(0.022) # Moment of inertia kg * m^2
        self.g = torch.tensor(9.81) # Gravity, m/s^2
        self.l = torch.tensor(0.225)  # Length of the quadrotor arm, m

        # In two_thrusters mode, control inputs are specified by the 
        # thrust (in N) provided by the right (u1) thruster and the left (u2) 
        # thruster. In differential mode, the control inputs are specified by 
        # the total thrust u1 (in N) and differential thrust u2 (in N)
        self.mode = mode 


    def forward(self, state, u, dt): 
        """
        This function will calculate the next state of the quadrotor given the current state, 
        control input, and time step. 
        
        :param state: The current state of the quadrotor 

        :param u: The control input. If u is a tensor, it is assumed to be a 2x1 tensor. if u is callable, 
        it is assumed to be a function that takes the current state as input and returns a 2x1 tensor.

        :param dt: The time step 
        
        :return: The next state of the quadrotor 
        """

        # Unpack the state and control input 
        x, y, theta, x_dot, y_dot, theta_dot = state

        # Check to see if the control input is a function
        if callable(u):
            u = u(state)
        
        u1, u2 = u

        # Calculate the next state 
        x_next = x + x_dot * dt
        y_next = y + y_dot * dt
        theta_next = theta + theta_dot * dt


        # Define net thrust and net torque variables 
        net_thrust = None 
        net_torque = None

        if self.mode == "two_thrusters": 
            net_thrust = u1 + u2
            net_torque = (u1 - u2) * self.l

        elif self.mode == "differential":
            net_thrust = u1
            net_torque = u2 * self.l

        # Calculate the next state
        x_dot_next = x_dot - (torch.sin(theta) * net_thrust / self.m) * dt
        y_dot_next = y_dot + (torch.cos(theta) * net_thrust / self.m - self.g) * dt
        theta_dot_next = theta_dot + net_torque / self.I * dt

        # Return the next state 
        return torch.tensor([x_next, y_next, theta_next, x_dot_next, y_dot_next, theta_dot_next])
    
    def rollout(self, initial_state, controller, dt, max_time=None, stop_condition=None): 
        """
        Perform a rollout of the quadrotor dynamics. 

        initial_state: torch.Tensor
            The initial state of the quadrotor.
        controller: callable, torch.Tensor -> torch.Tensor
            The controller for the quadrotor. This should be a function that takes the current state of the quadrotor
            as input and returns the control input as output.
        dt: float, timestep duration
            The duration of each timestep.
        max_time: float, optional, maximum simulation time 
        stop_condition: callable, optional, stop condition for the simulation
            The stop condition for the simulation. If provided, the simulation will stop when this condition is met. 
        """

        state = initial_state
        states = [state]
        time_elapsed = 0.0
        times = [time_elapsed]

        while True:
            if max_time is not None and time_elapsed >= max_time: 
                break

            if stop_condition is not None and stop_condition(state): 
                break

            control_input = controller(state)
            
            # Calculate the next state
            state = self.forward(state, control_input, dt)
            states.append(state)

            # Update the time
            time_elapsed += dt
            times.append(time_elapsed)

        return torch.stack(states), torch.tensor(times)
