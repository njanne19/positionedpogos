import math 
import numpy as np 
from pydrake.all import (
    LinearQuadraticRegulator, 
    Linearize, 
    MultibodyPlant, 
    System, 
    DiagramBuilder, 
    ConstantVectorSource, 
    Gain, 
    Adder, 
    Simulator
)

from underactuated.quadrotor2d import Quadrotor2D, Quadrotor2DVisualizer

class ClosedLoopPlanarQuadrotor(): 
    
    def __init__(self, controller_constructor: System): 
        
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
        
        # Then connect the output of the controller and connect it to the input of the plant 
        self.builder.Connect(
            self.controller.get_output_port(0), 
            self.plant.get_input_port(0)
        )
        
        # Next setup the setpoint system 
        self.setpoint_vector = self.builder.AddNamedSystem("Setpoint Vector", ConstantVectorSource(np.zeros([6, 1]))) 
        self.setpoint_inverter = self.builder.AddNamedSystem("Inverter", Gain(-1, 6)) 
        self.setpoint_adder = self.builder.AddNamedSystem("Error Calc", Adder(2, 6)) 
        
        # Connect setpoint logic together
        self.builder.Connect(
            self.setpoint_vector.get_output_port(0), 
            self.setpoint_inverter.get_input_port(0)
        )
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
        
        # Then build the diagram after everything is fully configured. 
        self.diagram = self.builder.Build() 
        self.diagram.set_name("ClosedLoopPlanarQuadrotor")
        
        # Also set up simulator here to be used later
        self.simulator = Simulator(self.diagram) 
        self.sim_context = self.simulator.get_mutable_context() 
        self.setpoint_context = self.diagram.GetMutableSubsystemContext(
            self.setpoint_vector, self.sim_context
        )
        
        # Establish time reference 
        self.time = 0.0 
        
        # For now, start at some random state
        self.sim_context.SetContinuousState(np.random.randn(6, 1))
               
        # Establish data logging variables. 
        self.data_log = {"time": [], "state": [], "setpoint": []}
        
        # Put first values on stack 
        self.record_current_state()

        # Finally, print out all context information: 
        print("ClosedLoopPlanarQuadrotor Instance") 
        print(self.sim_context) 
        print(f"Total Stats: {self.sim_context.num_total_states()}")
        print(f"Current Mutable State: {self.sim_context.get_mutable_state()}")
        print(f"Setpoint Mutable Context: {self.setpoint_context}")
        print(f"Setpoint Mutable State: {self.setpoint_context.get_mutable_state()}")
        

    def record_current_state(self): 
        self.data_log["time"].append(self.time)
        self.data_log["state"].append(self.sim_context.get_continuous_state_vector().CopyToVector())
        self.data_log["setpoint"].append(self.setpoint_vector.get_source_value(self.setpoint_context).CopyToVector())
        
    def get_data_log(self): 
        for key in self.data_log.keys(): 
            if isinstance(self.data_log[key], list): 
                self.data_log[key] = np.array(self.data_log[key])
        return self.data_log
        
    def step(self, dt, setpoint_update = None): 
        
        # Update the setpoint vector if applicable
        if setpoint_update is not None:
            self.setpoint_context.FixInputPort(0, setpoint_update)
        
        # Step the simulator forward 
        self.simulator.AdvanceTo(self.time + dt)
        
        # Update the time 
        self.time += dt
        
        # Record the updated state
        self.record_current_state()