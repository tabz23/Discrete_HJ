# environments/dubins_car.py
import numpy as np
from typing import List, Tuple
from .base import Environment

class DubinsCarEnvironment(Environment):
    """Dubins car with constant velocity and static obstacle"""
    
    def __init__(self, 
                 v_const: float = 1.0,
                 dt: float = 0.1,
                 hazard_position: np.ndarray = np.array([0.0, 0.0]),
                 hazard_size: float = 1.0,
                 state_bounds: List[Tuple[float, float]] = None):
        
        self.v_const = v_const
        self.dt = dt
        self.hazard_position = hazard_position
        self.hazard_size = hazard_size
        
        if state_bounds is None:
            self.state_bounds = [(-3.0, 3.0), (-3.0, 3.0), (-np.pi, np.pi)]
        else:
            self.state_bounds = state_bounds
            
    def dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Dubins car dynamics
        state: [x, y, theta]
        action: 0 (left: dtheta = -1), 1 (straight: dtheta = 0), 2 (right: dtheta = 1)
        """
        x, y, theta = state
        
        # Action to angular velocity mapping
        dtheta_map = {-1: 0, 0: 1, 1: 2}  # Map to your specified actions
        if action == 0:  # turn left
            dtheta = -1.0
        elif action == 1:  # straight
            dtheta = 0.0
        else:  # turn right
            dtheta = 1.0
            
        # Euler integration
        x2 = x + self.v_const * np.cos(theta) * self.dt
        y2 = y + self.v_const * np.sin(theta) * self.dt  
        theta2 = theta + dtheta * self.dt
        
        # Normalize angle to [-pi, pi]
        theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
        
        return np.array([x2, y2, theta2])
    
    def failure_function(self, state: np.ndarray) -> float:
        """Distance to hazard (negative inside obstacle)"""
        x, y, _ = state
        pos = np.array([x, y])
        h_val = np.linalg.norm(pos - self.hazard_position)
        return h_val - self.hazard_size
    
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        return self.state_bounds
    
    def get_actions(self) -> List[int]:
        return [0, 1, 2]  # left, straight, right