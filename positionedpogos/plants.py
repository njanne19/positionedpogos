import math 
import numpy as np 
from pydrake.systems.framework import System, DiagramBuilder, BasicVector
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import ( 
    Linearize, 
    Adder, 
    Gain, 
    ConstantVectorSource, 
    Saturation
)
from pydrake.systems.controllers import (
    LinearQuadraticRegulator
)
from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

class ClosedLoopPlanarQuadrotor(): 
    
    def __init__(self, controller_constructor: System, initial_state = None, initial_setpoint = None): 
        
        # We want a larger "system" object 
        # we can accomplish this by importing our plant
        # controller, etc. separately, and building 
        # with the diagram builder. 
        self.builder = DiagramBuilder() 
        self.plant = self.builder.AddNamedSystem("Quadrotor", Quadrotor2D())
        
        # Take in an already initialized controller 
        # and use it here 
        self.controller_constructor = controller_constructor
        self.controller = self.builder.AddNamedSystem("Controller", self.controller_constructor(self.plant))
        
        # Next setup the setpoint system 
        self.setpoint_inverter = self.builder.AddNamedSystem("Inverter", Gain(-1, 6)) 
        self.setpoint_adder = self.builder.AddNamedSystem("Error Calc", Adder(2, 6)) 
        
        # Add saturation 
        self.saturation = self.builder.AddSystem(Saturation(min_value=[0, 0], max_value=[15, 15]))
        
        # Connect setpoint logic together
        self.builder.Connect(
            self.setpoint_inverter.get_output_port(0),
            self.setpoint_adder.get_input_port(1),
        )
        self.builder.Connect(
            self.plant.get_output_port(0),
            self.setpoint_adder.get_input_port(0)
        )      
        self.builder.Connect(
            self.setpoint_adder.get_output_port(0),
            self.controller.get_input_port(0)  
        )
        
        # Saturate output of controller 
        self.builder.Connect(
            self.controller.get_output_port(0),
            self.saturation.get_input_port(0)
        )
        
        # Then connect the output of the saturation and connect it to the input of the plant 
        self.builder.Connect(
            self.saturation.get_output_port(0), 
            self.plant.get_input_port(0)
        )
        
        # Then build the diagram after everything is fully configured. 
        self.diagram = self.builder.Build() 
        self.diagram.set_name("ClosedLoopPlanarQuadrotor")
        
        # Also set up simulator here to be used later
        self.simulator = Simulator(self.diagram) 
        self.sim_context = self.simulator.get_mutable_context() 
        
        
        # Generate additional context objects for data logging 
        self.inverter_context = self.diagram.GetMutableSubsystemContext(
            self.setpoint_inverter, self.sim_context
        )
        self.plant_context = self.diagram.GetMutableSubsystemContext(
            self.plant, self.sim_context
        )
        self.error_context = self.diagram.GetMutableSubsystemContext(
            self.setpoint_adder, self.sim_context
        ) 
        
        # Establish time reference 
        self.time = 0.0 
        
        # For now, start at some random state
        if initial_state is not None: 
            self.sim_context.SetContinuousState(initial_state)
        else: 
            self.sim_context.SetContinuousState(np.random.randn(6, 1))
        
        # Set up setpoint itself: 
        if initial_setpoint is not None: 
            self.setpoint_vector = self.setpoint_inverter.get_input_port(0).FixValue(self.inverter_context, initial_setpoint)
        else: 
            self.setpoint_vector = self.setpoint_inverter.get_input_port(0).FixValue(self.inverter_context, np.array([0, 0, 0, 0, 0, 0]))
               
        # Establish data logging variables. 
        self.data_log = {"time": [], "state": [], "setpoint": [], "input": [], "error": []}
        
        # Put first values on stack 
        self.record_current_state()

        # Finally, print out all context information: 
        # print("ClosedLoopPlanarQuadrotor Instance") 
        # print(self.sim_context) 
        # print(f"Total Stats: {self.sim_context.num_total_states()}")
        # print(f"Current Mutable State: {self.sim_context.get_mutable_state()}")
        

    def record_current_state(self): 
        self.data_log["time"].append(self.time)
        self.data_log["state"].append(self.sim_context.get_continuous_state_vector().CopyToVector())
        self.data_log["setpoint"].append(self.setpoint_vector.GetMutableData().get_value().CopyToVector())
        self.data_log["input"].append(self.plant.get_input_port(0).Eval(self.plant_context))
        self.data_log["error"].append(self.setpoint_adder.get_output_port(0).Eval(self.error_context))
        
    def get_current_state(self): 
        data = {
            "time": self.time, 
            "state": self.sim_context.get_continuous_state_vector().CopyToVector(),
            "setpoint": self.setpoint_vector.GetMutableData().get_value().CopyToVector(),
            "input": self.plant.get_input_port(0).Eval(self.plant_context),
            "error": self.setpoint_adder.get_output_port(0).Eval(self.error_context)
        }
        return data
        
    def get_data_log(self): 
        for key in self.data_log.keys(): 
            if isinstance(self.data_log[key], list): 
                self.data_log[key] = np.array(self.data_log[key])
        return self.data_log
        
        
    def update_setpoint(self, new_setpoint): 
        self.setpoint_vector.GetMutableData().set_value(new_setpoint)
        
    def update_initial_state(self, new_state): 
        self.sim_context.SetContinuousState(new_state)
        
    def step(self, dt, new_setpoint = None): 
        
        # Update the setpoint vector if applicable
        if new_setpoint is not None: 
            self.update_setpoint(new_setpoint)
        
        # Step the simulator forward 
        self.simulator.AdvanceTo(self.time + dt)
        
        # Update the time 
        self.time += dt
        
        # Record the updated state
        self.record_current_state()