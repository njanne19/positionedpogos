import matplotlib.pyplot as plt
import pickle
import sys


def main(filename): 
    figx = pickle.load(open(filename, "rb"))
    figx.show()
    
if __name__ == "__main__":
   
    # Accept one argument, the path to a pickle file you want to open
    if len(sys.argv) != 2: 
        print("Usage: python view_3d_plot.py <path_to_pickle>")
        sys.exit(1)
    else: 
        main(sys.argv[1])