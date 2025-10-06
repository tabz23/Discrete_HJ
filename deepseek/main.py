# main.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from environments.dubins_car import DubinsCarEnvironment
from environments.base import Environment
from discretization.tree import TreeNode
from algorithms.algorithm1 import Algorithm1
from algorithms.algorithm3 import Algorithm3

def create_initial_discretization(environment: Environment, n_splits: int = 2) -> List[TreeNode]:
    """Create initial tree discretization"""
    bounds = environment.get_state_bounds()
    root = TreeNode(bounds)
    
    # Simple initial splitting
    current_level = [root]
    for _ in range(n_splits):
        next_level = []
        for node in current_level:
            next_level.extend(node.split())
        current_level = next_level
    
    return current_level

def main():
    # Create Dubins car environment
    env = DubinsCarEnvironment(
        v_const=1.0,
        dt=0.1,
        hazard_position=np.array([0.0, 0.0]),
        hazard_size=1.0
    )
    
    # Create initial discretization
    initial_tree = create_initial_discretization(env, n_splits=2)
    
    # Choose algorithm
    algorithm_choice = input("Choose algorithm (1 for Discretization, 3 for Adaptive Refinement): ")
    
    if algorithm_choice == "1":
        # Lipschitz constants (you'll need to derive these properly)
        L_f = 1.0  # Example - derive properly for your system
        L_l = 1.0  # Example - derive properly for your system
        
        algorithm = Algorithm1(
            environment=env,
            initial_discretization=initial_tree,
            L_f=L_f,
            L_l=L_l,
            gamma=0.95
        )
        
    else:  # Algorithm 3
        L_f = 1.0
        L_l = 1.0
        
        algorithm = Algorithm3(
            environment=env,
            initial_tree=initial_tree,
            L_f=L_f,
            L_l=L_l,
            gamma=0.95,
            refinement_threshold=0.1
        )
    
    # Run the algorithm
    final_discretization = algorithm.run(max_iterations=50, tolerance=1e-4)
    
    print(f"Final discretization has {len(final_discretization)} cells")
    print("Algorithm completed successfully!")

if __name__ == "__main__":
    main()