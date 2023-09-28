from positionedpogos.controllers import QuadrotorLQR
from positionedpogos.plants import ClosedLoopPlanarQuadrotor
from positionedpogos.visualizations import visualize_sim

def main(): 
    
    clpqr = ClosedLoopPlanarQuadrotor(QuadrotorLQR)
    
    dt = 0.1 
    while (clpqr.time < 5): 
        clpqr.step(dt) 
        
    print(clpqr.get_data_log())
    visualize_sim(clpqr) 
    
if __name__ == "__main__": 
    main() 