# environments/base.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

class Environment(ABC):
    """Base environment class for defining dynamics and safety properties"""
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action: int) -> np.ndarray:
        """Compute next state given current state and action"""
        pass
    
    @abstractmethod
    def failure_function(self, state: np.ndarray) -> float:
        """
        Failure function l(x)
        Returns: negative if unsafe (inside obstacle), positive if safe
        """
        pass
    
    @abstractmethod
    def get_state_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for each state dimension"""
        pass
    
    @abstractmethod
    def get_actions(self) -> List[int]:
        """Get available actions"""
        pass
    
    def is_safe(self, state: np.ndarray) -> bool:
        """Check if state is safe"""
        return self.failure_function(state) >= 0