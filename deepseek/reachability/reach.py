# reachability/reach.py
import numpy as np
from typing import List, Set
from discretization.tree import TreeNode
from environments.base import Environment

class ReachabilityAnalyzer:
    """Compute reachable sets using the tree discretization"""
    
    def __init__(self, environment: Environment):
        self.env = environment
    
    def compute_reachable_set(self, 
                            node: TreeNode, 
                            action: int,
                            tree_leaves: List[TreeNode]) -> Set[TreeNode]:
        """
        Compute reachable set from a given cell for a given action
        Returns set of leaf nodes that are reachable
        """
        reachable_nodes = set()
        
        # Sample points in the cell (center and vertices)
        sample_points = self._sample_cell(node)
        
        for point in sample_points:
            next_state = self.env.dynamics(point, action)
            
            # Find which leaf node contains the next state
            for leaf in tree_leaves:
                if leaf.contains(next_state):
                    reachable_nodes.add(leaf)
                    break
        
        return reachable_nodes
    
    def _sample_cell(self, node: TreeNode, n_samples: int = 8) -> List[np.ndarray]:
        """Sample points within a cell"""
        samples = []
        bounds = node.bounds
        
        # Always include center
        samples.append(node.get_center())
        
        # Sample additional points (simplified - in practice you'd want more sophisticated sampling)
        for _ in range(n_samples - 1):
            point = []
            for min_val, max_val in bounds:
                point.append(np.random.uniform(min_val, max_val))
            samples.append(np.array(point))
            
        return samples