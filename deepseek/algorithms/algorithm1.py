# algorithms/algorithm1.py
import numpy as np
from typing import List, Dict, Tuple
from discretization.tree import TreeNode
from environments.base import Environment
from reachability.reach import ReachabilityAnalyzer

class Algorithm1:
    """Discretization Routine (Algorithm 1)"""
    
    def __init__(self, 
                 environment: Environment,
                 initial_discretization: List[TreeNode],
                 L_f: float,
                 L_l: float,
                 gamma: float = 0.95):
        
        self.env = environment
        self.tree_leaves = initial_discretization
        self.L_f = L_f  # Lipschitz constant for dynamics
        self.L_l = L_l  # Lipschitz constant for failure function
        self.gamma = gamma
        self.reach_analyzer = ReachabilityAnalyzer(environment)
        
    def run(self, max_iterations: int = 100, tolerance: float = 1e-4):
        """Run the discretization algorithm"""
        V_prev = np.zeros(len(self.tree_leaves))
        
        for iteration in range(max_iterations):
            V_current = self._value_iteration_step()
            
            # Compute infinity norm difference
            diff = np.max(np.abs(V_current - V_prev))
            print(f"Iteration {iteration}: ||V_k - V_{iteration-1}||∞ = {diff}")
            
            # Check contraction
            if diff < tolerance:
                print(f"Converged after {iteration} iterations")
                break
                
            V_prev = V_current
            
            # Plot value function (you would implement this separately)
            self._plot_value_function(iteration)
        
        return self.tree_leaves
    
    def _value_iteration_step(self) -> np.ndarray:
        """Perform one step of value iteration"""
        new_values = np.zeros(len(self.tree_leaves))
        
        for i, leaf in enumerate(self.tree_leaves):
            best_value = -np.inf
            
            for action in self.env.get_actions():
                # Compute reachable set
                reachable_nodes = self.reach_analyzer.compute_reachable_set(
                    leaf, action, self.tree_leaves)
                
                if not reachable_nodes:
                    continue
                    
                # Compute expected value
                expected_value = 0.0
                # Simplified: uniform distribution over reachable nodes
                prob = 1.0 / len(reachable_nodes) if reachable_nodes else 0.0
                
                for node in reachable_nodes:
                    node_idx = self.tree_leaves.index(node)
                    # Reward: 0 if safe, -1 if unsafe
                    reward = 0.0 if self.env.is_safe(node.get_center()) else -1.0
                    expected_value += prob * (reward + self.gamma * node.value)
                
                if expected_value > best_value:
                    best_value = expected_value
            
            new_values[i] = best_value if best_value > -np.inf else 0.0
        
        # Update values
        for i, leaf in enumerate(self.tree_leaves):
            leaf.value = new_values[i]
            
        return new_values
    
    def _plot_value_function(self, iteration: int):
        """Plot and save value function (simplified - implement based on your needs)"""
        # This would use matplotlib to plot the value function
        # and save to a file
        pass