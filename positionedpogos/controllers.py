import math 
import numpy as np 
from pydrake.systems.framework import System
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.primitives import Linearize
from pydrake.systems.controllers import (
    LinearQuadraticRegulator
)
from pydrake.all import System

def QuadrotorLQR(plant: System): 
    
    context = plant.CreateDefaultContext() 
    context.SetContinuousState(np.zeros([6, 1]))
    plant.get_input_port(0).FixValue(
        context, plant.mass * plant.gravity / 2.0 * np.array([1, 1])
    )
    
    Q = np.diag([10, 10, 10, 1, 1, (plant.length /2.0 /np.pi)])
    R = np.array([[0.1, 0.05], [0.05, 0.1]]) 
    
    return LinearQuadraticRegulator(plant, context, Q, R) 