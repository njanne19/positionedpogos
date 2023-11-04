from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim
import numpy as np

def main(): 
    
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR)
    total_time = 20.0
    num_targets = 2
    target_interval = total_time / num_targets
    target_index = 0
    
    # Without angles 
    targets = [[np.random.rand()*2 - 1, np.random.rand()*2, 0, 0, 0, 0] for _ in range(num_targets)]
    # With angles
    # targets = [[np.random.rand()*2 - 1, np.random.rand()*2, np.random.rand()*np.deg2rad(10), 0, 0, 0] for _ in range(num_targets)]
    targets_set = [False for _ in range(num_targets)]
    
    dt = 0.1 
    while (clpqr.time < total_time): 
        
        if clpqr.time > target_interval * target_index and not targets_set[target_index]: 
            clpqr.update_setpoint(targets[target_index])
            targets_set[target_index] = True
            target_index += 1
        else: 
            clpqr.step(dt) 
        
    print("setpoint") 
    print(clpqr.get_data_log()["setpoint"])
    print(np.unique(clpqr.get_data_log()["setpoint"]))
    visualize_sim(clpqr) 
    
if __name__ == "__main__": 
    main() 