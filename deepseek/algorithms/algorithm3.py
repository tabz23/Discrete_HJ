# algorithms/algorithm3.py  
import numpy as np
from typing import List, Dict, Tuple
from discretization.tree import TreeNode
from environments.base import Environment
from reachability.reach import ReachabilityAnalyzer

class Algorithm3:
    """Adaptive refinement of boundary cells (Algorithm 3 with implementation details)"""
    
    def __init__(self,
                 environment: Environment,
                 initial_tree: List[TreeNode],
                 L_f: float,
                 L_l: float,
                 gamma: float = 0.95,
                 refinement_threshold: float = 0.1):
        
        self.env = environment
        self.tree_leaves = initial_tree
        self.L_f = L_f
        self.L_l = L_l  
        self.gamma = gamma
        self.refinement_threshold = refinement_threshold
        self.reach_analyzer = ReachabilityAnalyzer(environment)
        
    def run(self, max_iterations: int = 100, tolerance: float = 1e-4):
        """Run adaptive refinement algorithm"""
        V_prev = np.zeros(len(self.tree_leaves))
        
        for iteration in range(max_iterations):
            # Value iteration step
            V_current = self._value_iteration_step()
            
            # Compute infinity norm difference
            diff = np.max(np.abs(V_current - V_prev))
            print(f"Iteration {iteration}: ||V_k - V_{iteration-1}||∞ = {diff}")
            
            # Adaptive refinement step
            if iteration > 0:  # Allow some value iteration first
                self._adaptive_refinement()
            
            if diff < tolerance:
                print(f"Converged after {iteration} iterations")
                break
                
            V_prev = V_current
            self._plot_value_function(iteration)
        
        return self.tree_leaves
    
    def _value_iteration_step(self) -> np.ndarray:
        """Similar to Algorithm 1 but with boundary detection"""
        new_values = np.zeros(len(self.tree_leaves))
        
        for i, leaf in enumerate(self.tree_leaves):
            # Mark boundary cells based on value function gradient
            self._mark_boundary_cells(leaf)
            
            best_value = -np.inf
            for action in self.env.get_actions():
                reachable_nodes = self.reach_analyzer.compute_reachable_set(
                    leaf, action, self.tree_leaves)
                
                if not reachable_nodes:
                    continue
                    
                expected_value = 0.0
                prob = 1.0 / len(reachable_nodes)
                
                for node in reachable_nodes:
                    node_idx = self.tree_leaves.index(node)
                    reward = 0.0 if self.env.is_safe(node.get_center()) else -1.0
                    expected_value += prob * (reward + self.gamma * node.value)
                
                if expected_value > best_value:
                    best_value = expected_value
            
            new_values[i] = best_value if best_value > -np.inf else 0.0
        
        for i, leaf in enumerate(self.tree_leaves):
            leaf.value = new_values[i]
            
        return new_values
    
    def _mark_boundary_cells(self, leaf: TreeNode):
        """Mark cells as boundary based on value function changes"""
        # Simplified boundary detection - in practice you'd use Lipschitz conditions
        center = leaf.get_center()
        safety_margin = self.env.failure_function(center)
        
        # Mark as boundary if close to obstacle boundary or value changes rapidly
        leaf.is_boundary = (abs(safety_margin) < self.refinement_threshold or 
                           self._has_large_value_gradient(leaf))
    
    def _has_large_value_gradient(self, leaf: TreeNode) -> bool:
        """Check if cell has large value gradient with neighbors"""
        # Simplified - would need proper neighbor finding in tree
        return False
    
    def _adaptive_refinement(self):
        """Refine boundary cells"""
        new_leaves = []
        
        for leaf in self.tree_leaves:
            if leaf.is_boundary and leaf.depth < 5:  # Limit max depth
                children = leaf.split()
                # Initialize children values
                for child in children:
                    child.value = leaf.value
                    # Mark children as boundary initially
                    child.is_boundary = True
                new_leaves.extend(children)
            else:
                new_leaves.append(leaf)
        
        self.tree_leaves = new_leaves