from positionedpogos.src.controllers import QuadrotorLQR
from positionedpogos.src.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.src.visualizations import visualize_sim

def main(): 
    
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR)
    
    dt = 0.1 
    while (clpqr.time < 20): 
        clpqr.step(dt) 
        
    print(clpqr.get_data_log())
    visualize_sim(clpqr) 