# discretization/tree.py
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

class TreeNode:
    """Node in the discretization tree"""
    
    def __init__(self, 
                 bounds: List[Tuple[float, float]],
                 parent: Optional['TreeNode'] = None,
                 depth: int = 0):
        self.bounds = bounds  # List of (min, max) for each dimension
        self.parent = parent
        self.depth = depth
        self.children = []
        self.value = 0.0  # Value function for this cell
        self.is_boundary = False
        self.is_leaf = True
        
    def split(self) -> List['TreeNode']:
        """Split node into children (binary split along longest dimension)"""
        if not self.is_leaf:
            return self.children
            
        # Find dimension with largest range
        dim_ranges = [max_val - min_val for min_val, max_val in self.bounds]
        split_dim = np.argmax(dim_ranges)
        
        # Split along that dimension
        min_val, max_val = self.bounds[split_dim]
        mid_val = (min_val + max_val) / 2.0
        
        # Create two children
        child1_bounds = self.bounds.copy()
        child1_bounds[split_dim] = (min_val, mid_val)
        
        child2_bounds = self.bounds.copy()  
        child2_bounds[split_dim] = (mid_val, max_val)
        
        self.children = [
            TreeNode(child1_bounds, self, self.depth + 1),
            TreeNode(child2_bounds, self, self.depth + 1)
        ]
        
        self.is_leaf = False
        return self.children
    
    def get_center(self) -> np.ndarray:
        """Get center point of this cell"""
        return np.array([(min_val + max_val) / 2.0 for min_val, max_val in self.bounds])
    
    def contains(self, state: np.ndarray) -> bool:
        """Check if state is within this cell's bounds"""
        for i, (min_val, max_val) in enumerate(self.bounds):
            if not (min_val <= state[i] <= max_val):
                return False
        return True