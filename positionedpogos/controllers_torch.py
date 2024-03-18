import torch 
import torch.nn as nn 

class PIDController(nn.Module): 
    def __init__(self, Kp, Ki, Kd): 
        super(PIDController, self).__init__()
        self.Kp = Kp 
        self.Ki = Ki 
        self.Kd = Kd 

        # Terms initialization
        self.integral = 0 
        self.prev_error = 0

    def forward(self, setpoint, actual, dt): 
        """
        Calculate PID control signal 

        :param setpoint: Desired setpoint 
        :param actual: Actual measurement
        :param dt: Time step
        :return: The control signal 

        """

        error = setpoint - actual
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return output
    
    def reset(self): 
        """Reset the integral and derivative terms"""
        self.integral = 0 
        self.prev_error = 0


class CascadedPIDController(nn.Module): 
    """
    Implements a cascaded PID controller for a 2D planar quadrotor
    """

    def __init__(self, Kp_x, Ki_x, Kd_x, 
                Kp_y_outer, Ki_y_outer, Kd_y_outer,
                Kp_y_inner, Ki_y_inner, Kd_y_inner,
                Kp_theta, Ki_theta, Kd_theta): 
        super(CascadedPIDController, self).__init__()

        # Initialize the PID controllers
        self.pid_x_outer = PIDController(Kp_x, Ki_x, Kd_x)
        self.pid_y_outer = PIDController(Kp_y_outer, Ki_y_outer, Kd_y_outer)
        self.pid_y_inner = PIDController(Kp_y_inner, Ki_y_inner, Kd_y_inner)
        self.theta_inner = PIDController(Kp_theta, Ki_theta, Kd_theta)

        self.m = torch.tensor(1.5) # Mass, kg
        self.I = torch.tensor(0.022) # Moment of inertia kg * m^2
        self.g = torch.tensor(9.81) # Gravity, m/s^2
        self.l = torch.tensor(0.225)  # Length of the quadrotor arm, m

    def forward(self, setpoint, actual, dt):

        # Outer loop: compute setpoints for inner loop
        x_setpoint = setpoint[0]
        y_setpoint = setpoint[1]

        desired_theta_for_x = self.pid_x_outer(x_setpoint, actual[0], dt)
        desired_y_for_y = self.pid_y_outer(y_setpoint, actual[1], dt)

        # Inner loop: compute control signals
        theta_correction = self.theta_inner(desired_theta_for_x, actual[2], dt)
        y_correction = self.pid_y_inner(desired_y_for_y, actual[1], dt)

        # Combine control signals to total and differential thrust 
        u1, u2 = self.translate_to_thrust(theta_correction, y_correction)
        return torch.tensor([u1, u2])
    
    def translate_to_thrust(self, theta_correction, y_correction): 
        """
        Convert PID corrections into motor thrust commands.

        Parameters:
        - theta_correction: The desired change in orientation.
        - y_correction: The desired change in altitude.

        Returns:
        - u1: Thrust command for the right motor.
        - u2: Thrust command for the left motor.
        """

        hover_thrust = self.m * self.g / 2  # Thrust per motor to hover

        # Adjust total thrust based on y_correction
        # y_correction is considered as an additional force required; convert this to equivalent thrust
        total_thrust_correction = self.m * y_correction

        # Calculate differential thrust required for theta_correction
        # Assume a simple proportional relationship; could be refined with actual quadrotor dynamics
        differential_thrust = theta_correction * self.l

        # Calculate the thrust for each motor
        u1 = hover_thrust + total_thrust_correction / 2 + differential_thrust / 2
        u2 = hover_thrust + total_thrust_correction / 2 - differential_thrust / 2

        return u1, u2