"""
Safety Value Functions with Formal Guarantees - OPTIMIZED WITH SPATIAL INDEX
==================================================================================

Key optimizations:
1. Persistent worker pool (no recreation overhead)
2. NumPy arrays instead of serialized Cell objects
3. Index-based worker communication (minimal data transfer)
4. Efficient successor cache using indices
5. R-tree spatial index for O(log N) successor computation

Usage:
    python safety_value_function_fixed.py --algorithm 2 --conservative --delta-max 1e-6 --gamma 0.05 --dt 0.05
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Set
import argparse
import time
import os
from itertools import product
import matplotlib.colors as mcolors
from multiprocessing import Pool, cpu_count
from functools import partial
import rtree
from scipy.integrate import odeint

# ============================================================================
# PART 1: ENVIRONMENT DEFINITIONS
# ============================================================================
class Environment(ABC):
    """Abstract base class for dynamics environments."""
    
    def __init__(self, dt: float, tau: float, **kwargs):
        self.dt = dt
        self.tau = tau
        
        ratio = tau / dt
        n_steps = int(np.round(ratio))
        
        # Check if ratio is close to an integer (not using modulo!)
        if not np.isclose(ratio, n_steps, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"tau ({tau}) must be evenly divisible by dt ({dt}). "
                f"Current tau/dt = {ratio:.10f} (should be ~{n_steps}). "
                f"Try tau={dt * n_steps} or dt={tau / n_steps}"
            )
        
        self.n_steps = int(np.round(tau / dt))
        print(f"  Time discretization: {self.n_steps} steps of dt={dt}s over tau={tau}s")
    @abstractmethod
    def get_state_bounds(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_action_space(self) -> List:
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action) -> np.ndarray:
        pass
    
    # NEW: Multi-step dynamics for computing trajectory
    @abstractmethod
    def dynamics_multi_step(self, state: np.ndarray, action, duration: float, dt: float) -> List[np.ndarray]:
        """
        Integrate dynamics from state with constant action for 'duration' seconds,
        returning states at intervals of 'dt'.
        """
        pass
    
    @abstractmethod
    def failure_function(self, state: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def reward_function(self, state: np.ndarray) -> float:  ##changed for RA case
        """Return reward value at state. Positive if in target set."""  ##changed for RA case
        pass  ##changed for RA case
    
    @abstractmethod
    def get_lipschitz_constants(self) -> Tuple[float, float, float]:  ##changed for RA case - now returns (L_f, L_l, L_r)
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        pass

# ----------------------------------------------------------------------------
# Dubins Car Environment (Unchanged)
# ----------------------------------------------------------------------------

class DubinsCarEnvironment(Environment):
    def __init__(self, v_const: float = 1.0, dt: float = 0.1, tau: float = 1.0,
                 state_bounds: np.ndarray = None, obstacle_position: np.ndarray = None,
                 obstacle_radius: float = 0.5,
                 target_position: np.ndarray = None, target_radius: float = 0.5):  ##changed for RA case - added target params
                 
        
        # Validate before anything else
        ratio = tau / dt
        n_steps = int(np.round(ratio))
        
        # Check if ratio is close to an integer (not using modulo!)
        if not np.isclose(ratio, n_steps, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"tau ({tau}) must be evenly divisible by dt ({dt}). "
                f"Current tau/dt = {ratio:.10f} (should be ~{n_steps}). "
                f"Try tau={dt * n_steps} or dt={tau / n_steps}"
            )
        
        self.v_const = v_const
        self.dt = dt
        self.tau = tau
        self.n_steps = int(np.round(tau / dt))
        

        
        if state_bounds is None:
            self.state_bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [-np.pi, np.pi]])
        else:
            self.state_bounds = state_bounds
        
        if obstacle_position is None:
            self.obstacle_position = np.array([0.0, 0.0])
        else:
            self.obstacle_position = obstacle_position
            
        self.obstacle_radius = obstacle_radius
        
        # Target set parameters  ##changed for RA case
        if target_position is None:  ##changed for RA case
            self.target_position = np.array([2.5, 0])  ##changed for RA case
        else:  ##changed for RA case
            self.target_position = target_position  ##changed for RA case
            
        self.target_radius = target_radius  ##changed for RA case
        # Lipschitz constants
        self.L_f = v_const
        self.L_l = np.sqrt(2)
        self.L_r = np.sqrt(2)  ##changed for RA case - same as L_l since target is also a circle
        self.actions = [-1.0, 0.0, 1.0]
    
    def get_state_bounds(self) -> np.ndarray:
        return self.state_bounds
    
    def get_action_space(self) -> List:
        return self.actions
    
    def _dubins_ode(self, state: np.ndarray, t: float, action: float) -> np.ndarray:
        x, y, theta = state
        dx_dt = self.v_const * np.cos(theta)
        dy_dt = self.v_const * np.sin(theta)
        dtheta_dt = action
        return np.array([dx_dt, dy_dt, dtheta_dt])
    
    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        """Single step dynamics for duration tau"""
        t_span = [0, self.tau]
        solution = odeint(self._dubins_ode, state, t_span, args=(action,), atol=1e-12, rtol=1e-12)
        state_next = solution[-1]
        # Normalize theta to [-π, π]
        theta_next = np.arctan2(np.sin(state_next[2]), np.cos(state_next[2]))
        state_next[2] = theta_next
        return state_next
    
    def dynamics_multi_step(self, state: np.ndarray, action: float, duration: float, dt: float) -> List[np.ndarray]:
        """
        Integrate dynamics and return states at checkpoints.
        Returns states at times [dt, 2*dt, 3*dt, ..., duration]
        """
        n_steps = int(np.ceil(duration / dt))
        t_eval = np.linspace(0, duration, n_steps + 1)  # Include t=0
        
        solution = odeint(self._dubins_ode, state, t_eval, args=(action,), atol=1e-12, rtol=1e-12)
        
        # Normalize theta for all states
        states = []
        for s in solution[1:]:  # Skip initial state
            theta_normalized = np.arctan2(np.sin(s[2]), np.cos(s[2]))
            s[2] = theta_normalized
            states.append(s.copy())
        
        return states

    def failure_function(self, state: np.ndarray) -> float:
        pos = state[:2]
        dist_to_obstacle = np.linalg.norm(pos - self.obstacle_position)
        return (dist_to_obstacle - self.obstacle_radius)
    
    def reward_function(self, state: np.ndarray) -> float:  ##changed for RA case
        """Negative signed distance to target circle (positive inside target)"""  ##changed for RA case
        pos = state[:2]  ##changed for RA case
        dist_to_target = np.linalg.norm(pos - self.target_position)  ##changed for RA case
        return -(dist_to_target - self.target_radius)  ##changed for RA case
    
    def get_lipschitz_constants(self) -> Tuple[float, float, float]:  ##changed for RA case
        return self.L_f, self.L_l, self.L_r  ##changed for RA case
    
    def get_state_dim(self) -> int:
        return 3
# class DubinsCarEnvironment(Environment):
#     def __init__(self, v_const: float = 1.0, dt: float = 0.1, tau: float = 1.0,
#                  state_bounds: np.ndarray = None, obstacle_position: np.ndarray = None,
#                  obstacle_radius: float = 0.5,
#                  target_position: np.ndarray = None, target_radius: float = 0.5):  ##changed for RA case - added target params
                 
        
#         # Validate before anything else
#         ratio = tau / dt
#         n_steps = int(np.round(ratio))
        
#         # Check if ratio is close to an integer (not using modulo!)
#         if not np.isclose(ratio, n_steps, rtol=1e-9, atol=1e-9):
#             raise ValueError(
#                 f"tau ({tau}) must be evenly divisible by dt ({dt}). "
#                 f"Current tau/dt = {ratio:.10f} (should be ~{n_steps}). "
#                 f"Try tau={dt * n_steps} or dt={tau / n_steps}"
#             )
        
#         self.v_const = v_const
#         self.dt = dt
#         self.tau = tau
#         self.n_steps = int(np.round(tau / dt))
        

        
#         if state_bounds is None:
#             self.state_bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [-np.pi, np.pi]])
#         else:
#             self.state_bounds = state_bounds
        
#         if obstacle_position is None:
#             self.obstacle_position = np.array([0.0, 0.0])
#         else:
#             self.obstacle_position = obstacle_position
            
#         self.obstacle_radius = obstacle_radius
        
#         # Target set parameters  ##changed for RA case
#         if target_position is None:  ##changed for RA case
#             self.target_position = np.array([2.5, 0])  ##changed for RA case
#         else:  ##changed for RA case
#             self.target_position = target_position  ##changed for RA case
            
#         self.target_radius = target_radius  ##changed for RA case
#         # Lipschitz constants
#         self.L_f = v_const
#         self.L_l = np.sqrt(2)
#         self.L_r = 1.0  # Linear function of position  
#         self.actions = [-1.0, 0.0, 1.0]
    
#     def get_state_bounds(self) -> np.ndarray:
#         return self.state_bounds
    
#     def get_action_space(self) -> List:
#         return self.actions
    
#     def _dubins_ode(self, state: np.ndarray, t: float, action: float) -> np.ndarray:
#         x, y, theta = state
#         dx_dt = self.v_const * np.cos(theta)
#         dy_dt = self.v_const * np.sin(theta)
#         dtheta_dt = action
#         return np.array([dx_dt, dy_dt, dtheta_dt])
    
#     def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
#         """Single step dynamics for duration tau"""
#         t_span = [0, self.tau]
#         solution = odeint(self._dubins_ode, state, t_span, args=(action,), atol=1e-12, rtol=1e-12)
#         state_next = solution[-1]
#         # Normalize theta to [-π, π]
#         theta_next = np.arctan2(np.sin(state_next[2]), np.cos(state_next[2]))
#         state_next[2] = theta_next
#         return state_next
    
#     def dynamics_multi_step(self, state: np.ndarray, action: float, duration: float, dt: float) -> List[np.ndarray]:
#         """
#         Integrate dynamics and return states at checkpoints.
#         Returns states at times [dt, 2*dt, 3*dt, ..., duration]
#         """
#         n_steps = int(np.ceil(duration / dt))
#         t_eval = np.linspace(0, duration, n_steps + 1)  # Include t=0
        
#         solution = odeint(self._dubins_ode, state, t_eval, args=(action,), atol=1e-12, rtol=1e-12)
        
#         # Normalize theta for all states
#         states = []
#         for s in solution[1:]:  # Skip initial state
#             theta_normalized = np.arctan2(np.sin(s[2]), np.cos(s[2]))
#             s[2] = theta_normalized
#             states.append(s.copy())
        
#         return states

#     def failure_function(self, state: np.ndarray) -> float:
#         pos = state[:2]
#         dist_to_obstacle = np.linalg.norm(pos - self.obstacle_position)
#         return (dist_to_obstacle - self.obstacle_radius)
        
#     def reward_function(self, state: np.ndarray) -> float:
#         """
#         Reward function for reach-avoid with boundary target zone:
#         - Positive when OUTSIDE the inner box [-2.8, 2.8] × [-2.8, 2.8]
#         - Negative when inside the inner box
        
#         Returns negative signed distance to inner box boundary
#         (positive in the outer "safety zone")
#         """
#         x, y = state[0], state[1]
        
#         # Inner box boundaries (target is OUTSIDE this box)
#         inner_x_min, inner_x_max = -2.5, 2.5
#         inner_y_min, inner_y_max = -2.5, 2.5
        
#         # Distance from inner box boundaries (positive when inside inner box)
#         dist_to_inner_x_min = x - inner_x_min  # positive if x > -2.5
#         dist_to_inner_x_max = inner_x_max - x  # positive if x < 2.5
#         dist_to_inner_y_min = y - inner_y_min  # positive if y > -2.5
#         dist_to_inner_y_max = inner_y_max - y  # positive if y < 2.5
        
#         # Minimum distance to inner box boundary
#         min_dist_to_inner_boundary = min(
#             dist_to_inner_x_min, 
#             dist_to_inner_x_max,
#             dist_to_inner_y_min, 
#             dist_to_inner_y_max
#         )
        
#         # Return NEGATIVE of distance
#         # Positive when OUTSIDE [-2.5, 2.5]² (in the green safety zone)
#         # Negative when INSIDE [-2.5, 2.5]² (still need to reach safety)
#         return -min_dist_to_inner_boundary
#     def get_lipschitz_constants(self) -> Tuple[float, float, float]:  ##changed for RA case
#         return self.L_f, self.L_l, self.L_r  ##changed for RA case
    
#     def get_state_dim(self) -> int:
#         return 3
# ----------------------------------------------------------------------------
# ✈️ 3D Aircraft Evasion Environment (ODEINT-integrated)
# ----------------------------------------------------------------------------
class EvasionEnvironment(Environment):
    """3D evasion problem"""
    
    def __init__(self, v_const=1.0, dt=0.05, tau=1.0,
                 obstacle_position=(0.0, 0.0),
                 obstacle_radius=1.0, state_bounds=None,  target_position: np.ndarray = None, target_radius: float = 0.5):
        self.obstacle_position = np.array(obstacle_position)
        self.obstacle_radius = obstacle_radius
        self.v_const = v_const
        self.dt = dt  # Checkpoint interval
        self.tau = tau  # Control duration
        # Validate before anything else
        ratio = tau / dt
        n_steps = int(np.round(ratio))
        
        # Check if ratio is close to an integer (not using modulo!)
        if not np.isclose(ratio, n_steps, rtol=1e-9, atol=1e-9):
            raise ValueError(
                f"tau ({tau}) must be evenly divisible by dt ({dt}). "
                f"Current tau/dt = {ratio:.10f} (should be ~{n_steps}). "
                f"Try tau={dt * n_steps} or dt={tau / n_steps}"
            )

        if state_bounds is None:
            self.state_bounds = np.array([[-3.0, 3.0],##changed this 
                                          [-3.0, 3.0],##changed this
                                          [-np.pi, np.pi]])
        else:
            self.state_bounds = state_bounds
            
        # Target set parameters  ##changed for RA case
        if target_position is None:  ##changed for RA case
            self.target_position = np.array([2.5, 0])  ##changed for RA case
        else:  ##changed for RA case
            self.target_position = target_position  ##changed for RA case
            
        self.target_radius = target_radius  ##changed for RA case

        self.actions = np.linspace(-1.0, 1.0, 5)
        self.L_f = 1 + self.v_const
        self.L_l = np.sqrt(2)
        self.L_r = np.sqrt(2)  ##changed for RA case - same as L_l since target is also a circle


    def get_state_bounds(self) -> np.ndarray:
        return self.state_bounds

    def get_action_space(self) -> List:
        return self.actions.tolist()

    def _evasion_ode(self, state: np.ndarray, t: float, u: float) -> np.ndarray:
        x1, x2, x3 = state
        v = self.v_const
        dx1 = -v + v * np.cos(x3) + u * x2
        dx2 = v * np.sin(x3) - u * x1
        dx3 = -u
        return [dx1, dx2, dx3]

    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        """Single step dynamics for duration tau"""
        t_span = [0, self.tau]
        sol = odeint(self._evasion_ode, state, t_span, args=(action,), atol=1e-12, rtol=1e-12)
        next_state = sol[-1]
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        return next_state
    
    def dynamics_multi_step(self, state: np.ndarray, action: float, duration: float, dt: float) -> List[np.ndarray]:
        """
        Integrate dynamics and return states at checkpoints.
        Returns states at times [dt, 2*dt, 3*dt, ..., duration]
        """
        n_steps = int(np.ceil(duration / dt))
        t_eval = np.linspace(0, duration, n_steps + 1)  # Include t=0
        
        solution = odeint(self._evasion_ode, state, t_eval, args=(action,), atol=1e-12, rtol=1e-12)
        
        # Normalize theta for all states
        states = []
        for s in solution[1:]:  # Skip initial state
            theta_normalized = np.arctan2(np.sin(s[2]), np.cos(s[2]))
            s[2] = theta_normalized
            states.append(s.copy())
        
        return states

    def failure_function(self, state: np.ndarray) -> float:
        x1, x2, _ = state
        dx, dy = x1 - self.obstacle_position[0], x2 - self.obstacle_position[1]
        return np.sqrt(dx**2 + dy**2) - self.obstacle_radius
    
    def reward_function(self, state: np.ndarray) -> float:  ##changed for RA case
        """Negative signed distance to target circle (positive inside target)"""  ##changed for RA case
        pos = state[:2]  ##changed for RA case
        dist_to_target = np.linalg.norm(pos - self.target_position)  ##changed for RA case
        return -(dist_to_target - self.target_radius)  ##changed for RA case
    

    def get_lipschitz_constants(self) -> Tuple[float, float, float]:  ##changed for RA case
        return self.L_f, self.L_l, self.L_r  ##changed for RA case

    def get_state_dim(self) -> int:
        return 3


# ============================================================================
# PART 2: CELL STRUCTURE
# ============================================================================

class Cell:
    """Represents a hyperrectangular cell in discretized state space."""
    
    def __init__(self, bounds: np.ndarray, cell_id: int = 0):
        self.bounds = bounds
        self.cell_id = cell_id
        self.center = np.mean(bounds, axis=1)
        
        # CHANGED: Remove V_lower_conservative, we'll modify V_lower directly
        self.V_upper = None
        self.V_lower = None
        self.l_upper = None
        self.l_lower = None
        self.r_upper = None  ##changed for RA case
        self.r_lower = None  ##changed for RA case
        
        self.children = []
        self.is_leaf = True
        self.is_refined = False
        self.parent = None
        
    def get_range(self, dim: int) -> float:
        return self.bounds[dim, 1] - self.bounds[dim, 0]
    
    def get_max_range_dim(self) -> int:
        ranges = [self.get_range(j) for j in range(len(self.bounds))]
        return np.argmax(ranges)
    
    def get_max_range(self) -> float:
        dim = self.get_max_range_dim()
        return self.get_range(dim)
    
    def contains_point(self, point: np.ndarray) -> bool:
        for j in range(len(point)):
            if point[j] < self.bounds[j, 0] or point[j] > self.bounds[j, 1]:
                return False
        return True
    
    def intersects(self, other_bounds: np.ndarray) -> bool:
        for j in range(len(self.bounds)):
            if self.bounds[j, 1] < other_bounds[j, 0] or self.bounds[j, 0] > other_bounds[j, 1]:
                return False
        return True
    
    def split(self, next_id: int) -> Tuple['Cell', 'Cell']:
        dim = self.get_max_range_dim()
        mid = (self.bounds[dim, 0] + self.bounds[dim, 1]) / 2.0
        
        bounds1 = self.bounds.copy()
        bounds1[dim, 1] = mid
        bounds2 = self.bounds.copy()
        bounds2[dim, 0] = mid
        
        child1 = Cell(bounds1, next_id)
        child2 = Cell(bounds2, next_id + 1)
        
        child1.parent = self
        child2.parent = self
        
        self.children = [child1, child2]
        self.is_leaf = False
        
        return child1, child2


class CellTree:
    """Tree structure to manage cells with adaptive refinement and spatial indexing."""
    
    def __init__(self, initial_bounds: np.ndarray, initial_resolution: int = 10):
        self.root_bounds = initial_bounds
        self.dim = len(initial_bounds)
        self.next_id = 0
        self.leaves = []
        self.all_cells = []
        
        # Spatial index for fast intersection queries
        self.spatial_index = None
        self.cell_id_to_index = {}
        
        self._create_initial_grid(initial_resolution)
        self._build_spatial_index()
    
    def _create_initial_grid(self, resolution: int):
        ranges = []
        for j in range(self.dim):
            a, b = self.root_bounds[j]
            ranges.append(np.linspace(a, b, resolution + 1))
        
        indices = [range(resolution) for _ in range(self.dim)]
        for idx in product(*indices):
            bounds = np.zeros((self.dim, 2))
            for j in range(self.dim):
                bounds[j, 0] = ranges[j][idx[j]]
                bounds[j, 1] = ranges[j][idx[j] + 1]
            
            cell = Cell(bounds, self.next_id)
            self.next_id += 1
            self.leaves.append(cell)
            self.all_cells.append(cell)
    
    def _build_spatial_index(self):
        """Build R-tree spatial index for fast intersection queries."""
        print(f"  Building spatial index for {len(self.leaves)} cells...")
        start_time = time.time()
        # ============ ADD INTEGRITY CHECK AT START ============
        non_leaves_at_start = [c for c in self.leaves if not c.is_leaf]
        if non_leaves_at_start:
            print(f"    ⚠️  WARNING: {len(non_leaves_at_start)} non-leaf cells at START of _build_spatial_index!")
    # ======================================================
        
        import rtree
        
        # Create properties for 3D index
        p = rtree.index.Property()
        p.dimension = 3
        
        # Create new index with properties
        idx = rtree.index.Index(properties=p)
        self.cell_id_to_index = {}
        
        # Insert all leaf cells
        for i, cell in enumerate(self.leaves):
            # ============ ADD CHECK DURING ITERATION ============
            if not cell.is_leaf:
                print(f"    ⚠️  Inserting NON-LEAF Cell {cell.cell_id} into spatial index!")
            # ====================================================
        
            bbox = (
                cell.bounds[0, 0], cell.bounds[1, 0], cell.bounds[2, 0],
                cell.bounds[0, 1], cell.bounds[1, 1], cell.bounds[2, 1]
            )
            idx.insert(i, bbox)
            self.cell_id_to_index[cell.cell_id] = i
        
        self.spatial_index = idx
        elapsed = time.time() - start_time
        print(f"  ✓ Spatial index built in {elapsed:.2f}s")
        
    def get_leaves(self) -> List[Cell]:
        return self.leaves
    
    def refine_cell(self, cell: Cell):
        if not cell.is_leaf:
            return
        
        child1, child2 = cell.split(self.next_id)
        self.next_id += 2
        
        self.leaves.remove(cell)
        self.leaves.extend([child1, child2])
        self.all_cells.extend([child1, child2])
        
        cell.is_refined = True
    
    def rebuild_spatial_index(self):
        """Rebuild spatial index after refinements."""
        self._build_spatial_index()
    
    # def get_intersecting_cells(self, bounds: np.ndarray) -> List[Cell]:
    #     """
    #     Fast intersection query using spatial index.
    #     """
    #     if self.spatial_index is None:
    #         return [cell for cell in self.leaves if cell.intersects(bounds)]
        
    #     query_bbox = (
    #         bounds[0, 0], bounds[1, 0], bounds[2, 0],
    #         bounds[0, 1], bounds[1, 1], bounds[2, 1]
    #     )
        
    #     hit_indices = list(self.spatial_index.intersection(query_bbox))
    #     return [self.leaves[i] for i in hit_indices]
    def get_intersecting_cells(self, bounds: np.ndarray) -> List[Cell]:
        """
        Fast intersection query using spatial index (handles theta wrapping).
        """
        if self.spatial_index is None:
            return [cell for cell in self.leaves if cell.intersects(bounds)]
        
        # For theta dimension, we may need to query with wrapped versions
        theta_min, theta_max = bounds[2, 0], bounds[2, 1]
        
        # Collect all candidates from multiple queries
        all_candidates = set()
        
        # Query 1: Original bounds
        query_bbox = (
            bounds[0, 0], bounds[1, 0], theta_min,
            bounds[0, 1], bounds[1, 1], theta_max
        )
        hit_indices = list(self.spatial_index.intersection(query_bbox))
        all_candidates.update(hit_indices)
        
        # Query 2: Wrap theta by -2π (shift left)
        query_bbox_wrapped_neg = (
            bounds[0, 0], bounds[1, 0], theta_min - 2*np.pi,
            bounds[0, 1], bounds[1, 1], theta_max - 2*np.pi
        )
        hit_indices = list(self.spatial_index.intersection(query_bbox_wrapped_neg))
        all_candidates.update(hit_indices)
        
        # Query 3: Wrap theta by +2π (shift right)
        query_bbox_wrapped_pos = (
            bounds[0, 0], bounds[1, 0], theta_min + 2*np.pi,
            bounds[0, 1], bounds[1, 1], theta_max + 2*np.pi
        )
        hit_indices = list(self.spatial_index.intersection(query_bbox_wrapped_pos))
        all_candidates.update(hit_indices)
        
        return [self.leaves[i] for i in all_candidates]
        
    def get_num_leaves(self) -> int:
        return len(self.leaves)


# ============================================================================
# PART 3: REACHABILITY ANALYZER
# ============================================================================
class GronwallReachabilityAnalyzer:
    """
    Reachability using Grönwall's inequality with multiple checkpoints.
    """
    
    def __init__(self, env, use_infinity_norm: bool = True, debug_verify: bool = False):
        self.env = env
        self.use_infinity_norm = use_infinity_norm
        self.debug_verify = debug_verify
        
        L_f, _, _ = env.get_lipschitz_constants()
        self.L = L_f
        self.dt = env.dt      # Checkpoint interval
        self.tau = env.tau    # Control horizon
        
        # Number of checkpoints
        self.n_checkpoints = int(np.round(self.tau / self.dt))
        
        # Compute checkpoint times
        self.checkpoint_times = np.linspace(self.dt, self.tau, self.n_checkpoints)
        
        # Growth factors for each checkpoint
        self.growth_factors = [np.exp(self.L * t) for t in self.checkpoint_times]#[0.2 for t in self.checkpoint_times]#
        
        # Debug statistics
        self.debug_query_count = 0
        self.debug_mismatch_count = 0
        
        print(f"\nGrönwall Reachability Initialized:")
        print(f"  Lipschitz constant L = {self.L}")
        print(f"  Checkpoint interval dt = {self.dt}")
        print(f"  Control horizon τ = {self.tau}")
        print(f"  Number of checkpoints = {self.n_checkpoints}")
        print(f"  Checkpoint times: {[f'{t:.3f}' for t in self.checkpoint_times]}")
        print(f"  Growth factors: {[f'{gf:.6f}' for gf in self.growth_factors]}")
    
    # @staticmethod
    # def fix_angle_interval(left_angle: float, right_angle: float) -> Tuple[float, float]:
    # #to check , this doesnt necessarily normalize to [-π, π] Actual range: [-π, 3π] for example
    # #for example left_angle = 1.0 right_angle = 4.0 is returned as [1,4]
    # #since:    
    # # Step 1: if abs(4.0 - 1.0) < 0.001  →  if 3.0 < 0.001  →  FALSE ✗
    
    # # Step 2: while 4.0 < 1.0  →  FALSE 
    
    # # Step 3: if 3.0 >= 6.273  →  FALSE 
    
    # # Step 4: while (1.0 < -3.14 or 4.0 < -3.14)  →  FALSE 
    
    # # Step 5: while (1.0 > 9.42 or 4.0 > 9.42)  →  FALSE 
    
    # # Step 6: while (1.0 > 3.14 and 4.0 > 3.14)  →  FALSE 
    
    # #NOW SAY WE HAVE # Cell bounds (after fix_angle_interval): [1.0, 4.0] AND other cell Reachable bounds: [-3.0, -1.0]
    # #then need to Shift Reachable by +2π to know that they intersect
    # #[1.0, 4.0] and [-3.0 + 2π, -1.0 + 2π]=[3.28,5,28] then not they interesect
    #     """Normalize an angle interval to a canonical form."""
    #     if abs(right_angle - left_angle) < 0.001:
    #         left_angle = right_angle - 0.01
        
    #     while right_angle < left_angle:
    #         right_angle += 2 * np.pi
        
    #     if right_angle - left_angle >= 2 * np.pi - 0.01:
    #         return -np.pi, np.pi
        
    #     while left_angle < -np.pi or right_angle < -np.pi:
    #         left_angle += 2 * np.pi
    #         right_angle += 2 * np.pi
        
    #     while left_angle > 3 * np.pi or right_angle > 3 * np.pi:
    #         left_angle -= 2 * np.pi
    #         right_angle -= 2 * np.pi
        
    #     while left_angle > np.pi and right_angle > np.pi:
    #         left_angle -= 2 * np.pi
    #         right_angle -= 2 * np.pi
        
    #     return left_angle, right_angle
    
    # @staticmethod
    # def intervals_intersect(a_left: float, a_right: float, 
    #                       b_left: float, b_right: float) -> bool:
    #     """Check if two angle intervals intersect."""
    #     return not (a_right < b_left or b_right < a_left)
    @staticmethod
    def intervals_intersect(a_left: float, a_right: float, 
                        b_left: float, b_right: float) -> bool:
        """Check if two angle intervals intersect (handles wrapping)."""
        # Standard intersection
        if not (a_right < b_left or b_right < a_left):
            return True
        
        # Check with b wrapped by +2π
        if not (a_right < b_left + 2*np.pi or b_right + 2*np.pi < a_left):
            return True
        
        # Check with b wrapped by -2π
        if not (a_right < b_left - 2*np.pi or b_right - 2*np.pi < a_left):
            return True
        
        return False
    
    # def _check_theta_intersection(self, candidate: Cell, reach_bounds: np.ndarray) -> bool:
    #     """Helper: Check if theta dimension intersects."""
        
    #     ##added for debugging
    #     original_left = candidate.bounds[2, 0]
    #     original_right = candidate.bounds[2, 1]
    #     ##added for debugging
    #     cand_theta_left, cand_theta_right = self.fix_angle_interval(
    #         candidate.bounds[2, 0], 
    #         candidate.bounds[2, 1]
    #     )
    #     reach_theta_left = reach_bounds[2, 0]
    #     reach_theta_right = reach_bounds[2, 1]
        
    #     #        ##added for debugging
    #     # Log if fix_angle_interval changed anything
    #     if not np.isclose(original_left, cand_theta_left) or \
    #         not np.isclose(original_right, cand_theta_right):
    #         print(f"fix_angle_interval changed [{original_left:.3f}, {original_right:.3f}] "
    #           f"→ [{cand_theta_left:.3f}, {cand_theta_right:.3f}]")
    #     #        ##added for debugging
    
        
    #     return self.intervals_intersect(
    #         cand_theta_left, cand_theta_right,
    #         reach_theta_left, reach_theta_right
    #     )
    
    # def compute_successor_cells(self, cell, action, cell_tree) -> List[Cell]:
    #     """
    #     Find all cells that intersect the reachable set at ANY checkpoint.
        
    #     Process:
    #     1. Roll out dynamics from cell center for tau duration with dt intervals
    #     2. At each checkpoint, compute Grönwall circle/bounds
    #     3. Query cells intersecting those bounds
    #     4. Union all successor sets (avoiding duplicates)
    #     """
    #     center = cell.center
        
    #     # Compute cell radius
    #     if self.use_infinity_norm:
    #         r = 0.5 * cell.get_max_range()
    #     else:
    #         ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
    #         r = 0.5 * np.linalg.norm(ranges)
        
    #     # Roll out dynamics to get states at each checkpoint
    #     checkpoint_states = self.env.dynamics_multi_step(center, action, self.tau, self.dt)
        
    #     # Use set to track unique successor cell IDs (automatic deduplication)
    #     successor_cell_ids = set()
        
    #     # For each checkpoint: compute reachable bounds and query successors
    #     for i, state_at_checkpoint in enumerate(checkpoint_states):
    #         # Grönwall expansion at this checkpoint time
    #         growth_factor_i = self.growth_factors[i]
    #         expansion_i = r * growth_factor_i
            
    #         # Compute reachable bounds (Grönwall circle) at this checkpoint
    #         reach_bounds = np.zeros((self.env.get_state_dim(), 2))
            
    #         # x bounds
    #         reach_bounds[0, 0] = state_at_checkpoint[0] - expansion_i
    #         reach_bounds[0, 1] = state_at_checkpoint[0] + expansion_i
            
    #         # y bounds
    #         reach_bounds[1, 0] = state_at_checkpoint[1] - expansion_i
    #         reach_bounds[1, 1] = state_at_checkpoint[1] + expansion_i
            
    #         # θ bounds (with normalization)
    #         theta_lower = state_at_checkpoint[2] - expansion_i
    #         theta_upper = state_at_checkpoint[2] + expansion_i
    #         # theta_lower, theta_upper = self.fix_angle_interval(theta_lower, theta_upper)
    #         reach_bounds[2, 0] = theta_lower
    #         reach_bounds[2, 1] = theta_upper
            
    #         # Query spatial index for cells intersecting this checkpoint's bounds
    #         candidates = cell_tree.get_intersecting_cells(reach_bounds)
            
    #         # Filter by theta intersection and add to successor set
    #         for candidate in candidates:
    #             # if self._check_theta_intersection(candidate, reach_bounds):
    #             successor_cell_ids.add(candidate.cell_id)
        
    #     # Convert cell IDs back to Cell objects
    #     cell_id_to_cell = {c.cell_id: c for c in cell_tree.get_leaves()}
    #     successors = [cell_id_to_cell[cid] for cid in successor_cell_ids 
    #                  if cid in cell_id_to_cell]
        
    #     return successors
    
    def print_debug_summary(self):
        """Print debug verification summary."""
        if self.debug_verify:
            print(f"\n{'='*70}")
            print("DEBUG VERIFICATION SUMMARY")
            print(f"{'='*70}")
            print(f"Total successor queries verified: {self.debug_query_count}")
            print(f"Mismatches found: {self.debug_mismatch_count}")
            if self.debug_mismatch_count == 0:
                print("✓ All spatial index queries matched linear search exactly!")
            print(f"{'='*70}\n")


# ============================================================================
# PART 4: OPTIMIZED PARALLEL WORKERS
# ============================================================================

def _bellman_update_optimized(task, shared_data):
    """
    Optimized worker: uses numpy arrays instead of serialized objects.
    """
    cell_idx, succ_indices_by_action = task
    V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, r_upper_arr, r_lower_arr, gamma = shared_data  ##changed for RA case - added r arrays
    
    l_upper = l_upper_arr[cell_idx]
    l_lower = l_lower_arr[cell_idx]
    r_upper = r_upper_arr[cell_idx]  ##changed for RA case
    r_lower = r_lower_arr[cell_idx]  ##changed for RA case
    
    # Skip Bellman backup if this cell has no successors for any action
    #I COMMENTED THIS BECAUSE IF SUCC SET EMPTY, IT WAS NOT DOING THE BELLMAN OPERATOR AND JUST SKIPPING
    # if all(len(succ) == 0 for succ in succ_indices_by_action):
    #     return (cell_idx, V_upper_arr[cell_idx], V_lower_arr[cell_idx])

        
    best_min_val = -np.inf
    best_max_val = -np.inf

    for succ_indices in succ_indices_by_action:
        if len(succ_indices) == 0:
            action_lower = -np.inf
            action_upper = -np.inf
        else:
            lower_vals = V_lower_arr[succ_indices]
            upper_vals = V_upper_arr[succ_indices]

            action_lower = gamma * np.min(lower_vals)
            action_upper = gamma * np.max(upper_vals)

        best_min_val = max(best_min_val, action_lower)
        best_max_val = max(best_max_val, action_upper)

    new_V_lower = min(l_lower, max(r_lower, best_min_val))
    new_V_upper = min(l_upper, max(r_upper, best_max_val))

    return (cell_idx, new_V_upper, new_V_lower)


def _initialize_cell_worker(args):
    """
    Worker for parallel cell initialization.
    """
    cell_id, bounds, center, env, L_l, L_r = args  ##changed for RA case - added L_r
    
    # Compute failure function at center
    l_center = env.failure_function(center)
    
    # Compute reward function at center  ##changed for RA case
    r_center = env.reward_function(center)  ##changed for RA case
    
    # Compute radius
    ranges = bounds[:, 1] - bounds[:, 0]
    r = 0.5 * np.max(ranges)
    
    l_lower = l_center - L_l * r
    l_upper = l_center + L_l * r
    
    r_lower = r_center - L_r * r  ##changed for RA case
    r_upper = r_center + L_r * r  ##changed for RA case
    
    return (cell_id, l_lower, l_upper, r_lower, r_upper)  ##changed for RA case


def _compute_checkpoint_worker(task):
    """
    Worker: Compute checkpoint states only (ODE integration).
    Returns trajectory for later successor computation.
    """
    cell_id, center, action, env, tau, dt = task
    checkpoint_states = env.dynamics_multi_step(center, action, tau, dt)
    return (cell_id, action, checkpoint_states)

# ============================================================================
# PART 5: OPTIMIZED VALUE ITERATION (ALGORITHMS 1 & 3)
# ============================================================================

class SafetyValueIterator:
    """Implements Algorithm 1 & 3 with optimized parallelization and spatial indexing."""
    
    def __init__(self, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: GronwallReachabilityAnalyzer, output_dir: Optional[str] = None,
                 n_workers: Optional[int] = None, precompute_successors: bool = False,args=None):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.args=args
        if output_dir is None:
            rname = type(reachability).__name__
            param_suffix = (
                f"dynamics_{args.dynamics}_"
                f"gamma_{args.gamma:.3f}_"
                f"dt_{env.dt:.3f}_"
                f"tau_{env.tau:.3f}_"
                f"tol_{args.tolerance:.1e}_"
                f"eps_{args.epsilon:.3f}"
                f"vi-iterations_{args.vi_iterations}"
                f"conservative_{args.conservative}"
                f"delta-max_{args.delta_max}"
                f"init_resol_{args.initial_resolution}"
            )
            output_dir = os.path.join(
                "./results_RA/fixed_bellmannew/machineeps",
                f"{rname}_{param_suffix}"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.L_f, self.L_l, self.L_r = env.get_lipschitz_constants()  ##changed for RA case
        # if gamma * self.L_f >= 1:
        #     raise ValueError(f"Contraction condition violated: γL_f = {gamma * self.L_f} >= 1")
        
        self.refinement_phase = 0
        
        # Parallelization settings
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.precompute_successors_flag = precompute_successors
        self.successor_cache = {}
        self.pool = None
        
        print(f"Initialized with γ={gamma}, L_f={self.L_f}, L_l={self.L_l},  L_r={self.L_r}")

        print(f"Contraction factor: γL_f = {gamma * self.L_f:.4f}")
        print(f"Using {self.n_workers} parallel workers")
        
        if precompute_successors:
            self._precompute_all_successors()
    
    def __del__(self):
        """Cleanup worker pool on deletion"""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
    
    def _precompute_all_successors(self):
        """
        HYBRID: Parallel ODE integration + Sequential spatial index queries.
        Achieves O(N log N) complexity instead of O(N²).
        """
        print(f"\nPrecomputing successor sets with HYBRID approach...")
        leaves = self.cell_tree.get_leaves()
        actions = self.env.get_action_space()
        
        n_tasks = len(leaves) * len(actions)
        print(f"  Step 1: Computing {n_tasks} trajectories in parallel...")
        start_time = time.time()
        
        # STEP 1: Parallel ODE integration (expensive part)
        ode_tasks = []
        for cell in leaves:
            for action in actions:
                ode_tasks.append((
                    cell.cell_id, 
                    cell.center.copy(), 
                    action, 
                    self.env, 
                    self.reachability.tau, 
                    self.reachability.dt
                ))
        
        chunksize = max(1, len(ode_tasks) // (self.n_workers * 4))
        
        if self.pool is not None:
            trajectory_results = self.pool.map(_compute_checkpoint_worker, ode_tasks, chunksize=chunksize)
        else:
            with Pool(self.n_workers) as pool:
                trajectory_results = pool.map(_compute_checkpoint_worker, ode_tasks, chunksize=chunksize)
        
        ode_time = time.time() - start_time
        print(f"  ✓ Trajectories computed in {ode_time:.2f}s ({len(ode_tasks)/ode_time:.1f} tasks/s)")
        
        # STEP 2: Sequential successor queries using spatial index (fast!)
        print(f"  Step 2: Computing successors using spatial index...")
        query_start = time.time()
        
        # Build cell lookup
        cell_dict = {cell.cell_id: cell for cell in leaves}
        
        for cell_id, action, checkpoint_states in trajectory_results:
            cell = cell_dict[cell_id]
            
            # Compute cell radius
            if self.reachability.use_infinity_norm:
                r = 0.5 * cell.get_max_range()
            else:
                ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
                r = 0.5 * np.linalg.norm(ranges)
            
            # Track unique successor cell IDs
            successor_cell_ids = set()
            
            # For each checkpoint: compute reachable bounds and query spatial index
            for i, state_at_checkpoint in enumerate(checkpoint_states):
                growth_factor_i = self.reachability.growth_factors[i]
                expansion_i = r * growth_factor_i
                
                # Compute reachable bounds at this checkpoint
                reach_bounds = np.zeros((self.env.get_state_dim(), 2))
                reach_bounds[0, 0] = state_at_checkpoint[0] - expansion_i
                reach_bounds[0, 1] = state_at_checkpoint[0] + expansion_i
                reach_bounds[1, 0] = state_at_checkpoint[1] - expansion_i
                reach_bounds[1, 1] = state_at_checkpoint[1] + expansion_i
                
                # θ bounds (with normalization)
                if self.env.get_state_dim() == 3:
                    theta_lower = state_at_checkpoint[2] - expansion_i
                    theta_upper = state_at_checkpoint[2] + expansion_i
                    # theta_lower, theta_upper = self.reachability.fix_angle_interval(theta_lower, theta_upper)
                    reach_bounds[2, 0] = theta_lower
                    reach_bounds[2, 1] = theta_upper
                
                # USE SPATIAL INDEX: O(log N) instead of O(N)!
                candidates = self.cell_tree.get_intersecting_cells(reach_bounds)
                
                # Filter by theta intersection
                for candidate in candidates:
                    # if self.reachability._check_theta_intersection(candidate, reach_bounds):
                    successor_cell_ids.add(candidate.cell_id)

            
            # Store in cache
            key = (cell_id, action)
            self.successor_cache[key] = list(successor_cell_ids)
     
        query_time = time.time() - query_start
        elapsed = time.time() - start_time
        
        print(f"  ✓ Spatial queries completed in {query_time:.2f}s")
        print(f"  ✓ Total precomputation: {elapsed:.2f}s ({n_tasks/elapsed:.1f} tasks/s)")
        print(f"    Breakdown: {ode_time/elapsed*100:.1f}% ODE, {query_time/elapsed*100:.1f}% queries")
  
        print(f"\n{'='*70}")
        print("DETAILED CHECK: Cell 0, Action 0.0")
        print(f"{'='*70}")

        cell_0 = next(c for c in leaves if c.cell_id == 0)
        print(f"Cell 0 center: {cell_0.center}")
        print(f"Cell 0 bounds: {cell_0.bounds}")
        print(f"Cell 0 radius: {0.5 * cell_0.get_max_range()}")

        action = 0.0
        checkpoint_states = self.env.dynamics_multi_step(cell_0.center, action, self.reachability.tau, self.reachability.dt)

        print(f"\nCheckpoint states (showing every 2nd):")
        for i in range(0, len(checkpoint_states), 2):
            t = self.reachability.checkpoint_times[i]
            state = checkpoint_states[i]
            print(f"  t={t:.1f}: state = {state}")

        print(f"\nFinal checkpoint analysis:")
        final_state = checkpoint_states[-1]
        r = 0.5 * cell_0.get_max_range()
        expansion = r * self.reachability.growth_factors[-1]
        print(f"  Final state: {final_state}")
        print(f"  Expansion: {expansion:.3f}")

        reach_bounds = np.zeros((3, 2))
        reach_bounds[0, 0] = final_state[0] - expansion
        reach_bounds[0, 1] = final_state[0] + expansion
        reach_bounds[1, 0] = final_state[1] - expansion
        reach_bounds[1, 1] = final_state[1] + expansion

        theta_lower = final_state[2] - expansion
        theta_upper = final_state[2] + expansion
        # theta_lower, theta_upper = self.reachability.fix_angle_interval(theta_lower, theta_upper)
        reach_bounds[2, 0] = theta_lower
        reach_bounds[2, 1] = theta_upper

        print(f"  Reachable bounds:")
        print(f"    x ∈ [{reach_bounds[0,0]:.3f}, {reach_bounds[0,1]:.3f}]")
        print(f"    y ∈ [{reach_bounds[1,0]:.3f}, {reach_bounds[1,1]:.3f}]")
        print(f"    θ ∈ [{reach_bounds[2,0]:.3f}, {reach_bounds[2,1]:.3f}]")

        print(f"  Original cell bounds:")
        print(f"    x ∈ [{cell_0.bounds[0,0]:.3f}, {cell_0.bounds[0,1]:.3f}]")
        print(f"    y ∈ [{cell_0.bounds[1,0]:.3f}, {cell_0.bounds[1,1]:.3f}]")
        print(f"    θ ∈ [{cell_0.bounds[2,0]:.3f}, {cell_0.bounds[2,1]:.3f}]")

        # Check spatial index query
        candidates = self.cell_tree.get_intersecting_cells(reach_bounds)
        print(f"  Spatial index returned {len(candidates)} candidates")
        print(f"  Cell 0 in candidates? {cell_0.cell_id in [c.cell_id for c in candidates]}")
    
        
    def initialize_cells(self):
        """Initialize stage cost bounds for all cells (parallelized)."""
        leaves = self.cell_tree.get_leaves()
        
        if len(leaves) == 0:
            return
        
        print(f"  Initializing {len(leaves)} cells in parallel...")
        start_time = time.time()
        
        # Prepare tasks
        tasks = [
            (cell.cell_id, cell.bounds.copy(), cell.center.copy(), self.env, self.L_l, self.L_r)  ##changed for RA case
            for cell in leaves
        ]
        
        # Use persistent pool if available, otherwise create temporary one
        if self.pool is not None:
            results = self.pool.map(_initialize_cell_worker, tasks)
        else:
            with Pool(self.n_workers) as pool:
                results = pool.map(_initialize_cell_worker, tasks)
        
        # Apply results
        cell_dict = {cell.cell_id: cell for cell in leaves}
        for cell_id, l_lower, l_upper, r_lower, r_upper in results:  ##changed for RA case
            cell = cell_dict[cell_id]
            cell.l_lower = l_lower
            cell.l_upper = l_upper
            cell.r_lower = r_lower  ##changed for RA case
            cell.r_upper = r_upper  ##changed for RA case
            cell.V_lower = cell.l_lower
            cell.V_upper = cell.l_upper
            
        elapsed = time.time() - start_time
        print(f"  ✓ Initialized in {elapsed:.2f}s ({len(leaves)/elapsed:.1f} cells/s)")
    
    def initialize_new_cells(self, new_cells: List[Cell]):
        """Initialize new cells after refinement (parallelized)."""
        if len(new_cells) == 0:
            return
        
        print(f"  Initializing {len(new_cells)} new cells in parallel...")
        start_time = time.time()
        
        # Prepare tasks
        tasks = [
            (cell.cell_id, cell.bounds.copy(), cell.center.copy(), self.env, self.L_l, self.L_r)  ##changed for RA case
            for cell in new_cells
        ]
            
        # Use persistent pool if available, otherwise create temporary one
        if self.pool is not None:
            results = self.pool.map(_initialize_cell_worker, tasks)
        else:
            with Pool(self.n_workers) as pool:
                results = pool.map(_initialize_cell_worker, tasks)
        
        # Apply results
        cell_dict = {cell.cell_id: cell for cell in new_cells}
        for cell_id, l_lower, l_upper, r_lower, r_upper in results:  ##changed for RA case
            cell = cell_dict[cell_id]
            cell.l_lower = l_lower
            cell.l_upper = l_upper
            cell.r_lower = r_lower  ##changed for RA case
            cell.r_upper = r_upper  ##changed for RA case
            cell.V_lower = cell.l_lower
            cell.V_upper = cell.l_upper
        
        elapsed = time.time() - start_time
        print(f"  ✓ Initialized in {elapsed:.2f}s ({len(new_cells)/elapsed:.1f} cells/s)")
        
    def value_iteration(self, max_iterations: int = 1000, convergence_tol: float = 1e-3,
                plot_freq: int = 10, conservative_mode: bool = False, delta_max: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run value iteration until convergence (Algorithm 1 or 3).
        """
        
        self.initialize_cells()
        if not self.successor_cache:
            print("  Precomputing successor sets...")
            self._precompute_all_successors()
        
        # CREATE PERSISTENT POOL ONCE
        if self.pool is None:
            self.pool = Pool(self.n_workers)
        
        conv_history_upper = []
        conv_history_lower = []
        
        print(f"\nStarting OPTIMIZED PARALLEL value iteration (max {max_iterations} iterations)...")
        print(f"Conservative mode: {conservative_mode}")
        if conservative_mode:
            print(f"Conservative tolerance δ_max: {delta_max}")
            print(f"Conservative margin will be: ε_cons = (γ * δ_actual) / (1 - γ)")
        else:
            print(f"Convergence tolerance: {convergence_tol}")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        
        for iteration in range(max_iterations):
            leaves = self.cell_tree.get_leaves()
            prev_upper = {cell.cell_id: cell.V_upper for cell in leaves}
            prev_lower = {cell.cell_id: cell.V_lower for cell in leaves}
                        # ============ ADDED: Detailed debugging output for Phase 0 ============
            if self.refinement_phase == 0 and plot_freq > 1 and iteration+1 % plot_freq == 0:
                print(f"\n{'='*100}")
                print(f"DETAILED CELL STATE - Iteration {iteration}")
                print(f"{'='*100}")
                
                for cell in leaves[0:1]:
                    print(f"\n--- Cell {cell.cell_id} ---")
                    print(f"  Bounds: x=[{cell.bounds[0,0]:.6f}, {cell.bounds[0,1]:.6f}], "
                        f"y=[{cell.bounds[1,0]:.6f}, {cell.bounds[1,1]:.6f}], "
                        f"θ=[{cell.bounds[2,0]:.6f}, {cell.bounds[2,1]:.6f}]")
                    print(f"  Center: {cell.center}")
                    print(f"  l_lower: {cell.l_lower:.10f}")
                    print(f"  l_upper: {cell.l_upper:.10f}")
                    print(f"  V_lower: {cell.V_lower:.10f}")
                    print(f"  V_upper: {cell.V_upper:.10f}")
                    
                    # Show successors for each action
                    actions = self.env.get_action_space()
                    for action in actions:
                        cache_key = (cell.cell_id, action)
                        if cache_key in self.successor_cache:
                            succ_ids = self.successor_cache[cache_key]
                            print(f"\n  Action {action:+.3f} → {len(succ_ids)} successors:")
                            
                            if len(succ_ids) == 0:
                                print(f"    (No successors)")
                            else:
                                # Get successor cells
                                cell_id_to_cell = {c.cell_id: c for c in leaves}
                                for succ_id in succ_ids:
                                    if succ_id in cell_id_to_cell:
                                        succ = cell_id_to_cell[succ_id]
                                        print(f"    → Cell {succ.cell_id}:")
                                        print(f"       Bounds: x=[{succ.bounds[0,0]:.6f}, {succ.bounds[0,1]:.6f}], "
                                            f"y=[{succ.bounds[1,0]:.6f}, {succ.bounds[1,1]:.6f}], "
                                            f"θ=[{succ.bounds[2,0]:.6f}, {succ.bounds[2,1]:.6f}]")
                                        print(f"       l_lower: {succ.l_lower:.10f}, l_upper: {succ.l_upper:.10f}")
                                        print(f"       V_lower: {succ.V_lower:.10f}, V_upper: {succ.V_upper:.10f}")
                        else:
                            print(f"\n  Action {action:+.3f} → (No cache entry)")
                
                print(f"\n{'='*100}\n")
            # ============ END ADDED SECTION ============
                
            # Build compact numpy arrays for efficient parallel processing
            n_cells = len(leaves)
            cell_ids = np.array([c.cell_id for c in leaves])
            V_upper_arr = np.array([c.V_upper for c in leaves])
            V_lower_arr = np.array([c.V_lower for c in leaves])
            l_upper_arr = np.array([c.l_upper for c in leaves])
            l_lower_arr = np.array([c.l_lower for c in leaves])
            r_upper_arr = np.array([c.r_upper for c in leaves])  ##changed for RA case
            r_lower_arr = np.array([c.r_lower for c in leaves])  ##changed for RA case
            
            # Build id->index mapping
            id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
            
            # Prepare compact tasks with indices
            tasks = []
            for idx, cell in enumerate(leaves):
                succ_indices_by_action = []
                for action in self.env.get_action_space():
                    cache_key = (cell.cell_id, action)
                    if cache_key in self.successor_cache:
                        succ_ids = self.successor_cache[cache_key]
                        succ_indices = [id_to_idx[sid] for sid in succ_ids if sid in id_to_idx]
                    else:
                        succ_indices = []
                    succ_indices_by_action.append(succ_indices)
                tasks.append((idx, succ_indices_by_action))
            
                # Shared data for workers
                shared_data = (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, r_upper_arr, r_lower_arr, self.gamma)  ##changed for RA case
            
            # Parallel Bellman updates
            worker_func = partial(_bellman_update_optimized, shared_data=shared_data)
            results = self.pool.map(worker_func, tasks)
            
            # Apply updates
            for cell_idx, new_upper, new_lower in results:
                leaves[cell_idx].V_upper = new_upper
                leaves[cell_idx].V_lower = new_lower
            
            # Check convergence
            diff_upper = max(abs(leaves[i].V_upper - prev_upper[leaves[i].cell_id]) for i in range(n_cells))
            diff_lower = max(abs(leaves[i].V_lower - prev_lower[leaves[i].cell_id]) for i in range(n_cells))
            
            # Store convergence history
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            if conservative_mode:
                # ALGORITHM 3: Conservative stopping condition
                delta_k = min(leaves[i].V_lower - prev_lower[leaves[i].cell_id] for i in range(n_cells))
                
                # Detailed convergence metrics
                print(f"Iteration {iteration + 1:3d}: "
                    f"||V̄^k - V̄^k-1||_∞ = {diff_upper:12.20f}, "
                    f"||V_^k - V_^k-1||_∞ = {diff_lower:12.20f}, "
                    f"δ^k = {delta_k:12.20f}")
                
                # MODIFIED: Conservative stopping with V_upper check
                if delta_k >= -delta_max and diff_upper < convergence_tol:
                    print(f"\n" + "="*60)
                    print("✓ CONSERVATIVE STOPPING CONDITION MET (Algorithm 3)")
                    print("="*60)
                    print(f"  δ^k = {delta_k:.10e} ≥ -δ_max = -{delta_max:.10e}")
                    print(f"  ||V̄^k - V̄^k-1||_∞ = {diff_upper:.10e} < {convergence_tol}")
                    
                    delta_to_use = abs(delta_k)
                    epsilon_cons = (self.gamma * delta_to_use) / (1 - self.gamma)
                    
                    print(f"  ε_cons = ({self.gamma} * {delta_to_use:.10e}) / {1 - self.gamma} = {epsilon_cons:.10e}")
                    print(f"  Correcting V_lower for {len(leaves)} cells: V_lower ← V_lower - ε_cons - ε_machine")
                    
                    # NEW: Track statistics
                    eps_machine = np.finfo(float).eps
                    cells_near_zero = 0
                    cells_flipped_sign = 0
                    
                    for cell in leaves:
                        old_value = cell.V_lower
                        
                        # Apply conservative correction
                        cell.V_lower = cell.V_lower - epsilon_cons - eps_machine
                        
                        # Track cells affected by cancellation
                        if abs(old_value - epsilon_cons) <  eps_machine:
                            cells_near_zero += 1
                        
                        # Track sign flips due to eps_machine subtraction
                        if old_value - epsilon_cons > 0 and cell.V_lower <= 0:
                            cells_flipped_sign += 1
                    
                    print(f"  ✓ Conservative correction applied successfully")
                    print(f"  Machine epsilon ε_machine = {eps_machine:.3e}")
                    
                    if cells_near_zero > 0:
                        print(f"  ⚠️  {cells_near_zero} cells experienced near-cancellation")
                    if cells_flipped_sign > 0:
                        print(f"  ⚠️  {cells_flipped_sign} cells changed from (+) to (≤0) due to ε_machine")
                    break
            else:
                # ALGORITHM 1: Standard convergence
                print(f"Iteration {iteration + 1:3d}: "
                    f"||V̄^k - V̄^k-1||_∞ = {diff_upper:12.20f}, "
                    f"||V_^k - V_^k-1||_∞ = {diff_lower:12.20f}")
                
                if diff_upper < convergence_tol and diff_lower < convergence_tol:
                    print(f"\n" + "="*50)
                    print("✓ STANDARD CONVERGENCE ACHIEVED (Algorithm 1)")
                    print("="*50)
                    print(f"  ||V̄^k - V̄^k-1||_∞ = {diff_upper:.20e} < tolerance = {convergence_tol}")
                    print(f"  ||V_^k - V_^k-1||_∞ = {diff_lower:.20e} < tolerance = {convergence_tol}")
                    print(f"  Converged in {iteration + 1} iterations")
                    
                    # Save final plot
                    suffix = "_final"
                    if self.refinement_phase > 0:
                        filename = os.path.join(
                            self.output_dir, 
                            f"iteration_{iteration + 1:04d}_refinement_{self.refinement_phase:02d}{suffix}.png"
                        )
                    else:
                        filename = os.path.join(
                            self.output_dir, 
                            f"iteration_{iteration + 1:04d}{suffix}.png"
                        )
                    # plot_value_function(self.env, self.cell_tree, filename, iteration + 1)
                    break
            
            # Periodic plotting
            if plot_freq > 0 and (iteration) % plot_freq == 0:
                suffix = ""
                if self.refinement_phase >= 0:
                    filename = os.path.join(
                        self.output_dir, 
                        f"iteration_{iteration + 1:04d}_refinement_{self.refinement_phase:02d}{suffix}.png"
                    )
                else:
                    filename = os.path.join(
                        self.output_dir, 
                        f"iteration_{iteration + 1:04d}{suffix}.png"
                    )
                plot_value_function(self.env, self.cell_tree, filename, iteration + 1)
                print(f"  [Plot saved at iteration {iteration + 1}]")
                
        # Final iteration summary if max iterations reached
        if iteration == max_iterations - 1:
            print(f"\n" + "!"*50)
            print(f"⚠️  MAXIMUM ITERATIONS REACHED: {max_iterations}")
            print("!"*50)
            if conservative_mode:
                delta_k = min(leaves[i].V_lower - prev_lower[leaves[i].cell_id] for i in range(n_cells))
                print(f"  Final δ^k = {delta_k:.20e}")
            print(f"  Final ||V̄^k - V̄^k-1||_∞ = {diff_upper:.20e}")
            print(f"  Final ||V_^k - V_^k-1||_∞ = {diff_lower:.20e}")
            
            # Apply conservative correction even if max iterations reached
            if conservative_mode:
                # MODIFIED: Use min(|delta_k|, delta_max) when max iterations reached
                delta_to_use = abs(delta_k)#delta_to_use = min(abs(delta_k), delta_max) #if delta_k < 0 else delta_max
                epsilon_cons = (self.gamma * delta_to_use) / (1 - self.gamma)
                
                print(f"  Using δ_actual = {delta_to_use:.10e} for correction")
                print(f"  Applying conservative margin: ε_cons = {epsilon_cons:.10e}")
                for cell in leaves:
                    cell.V_lower = cell.V_lower - epsilon_cons
                print(f"  ✓ Conservative correction applied to {len(leaves)} cells")
                
        filename = os.path.join(
                self.output_dir, 
                f"value_function_phase_{0}_complete.png"
            )      
        plot_value_function(self.env, self.cell_tree, filename, iteration + 1)
        
        
        print(f"\nValue iteration completed in {iteration + 1} iterations")

        return np.array(conv_history_upper), np.array(conv_history_lower)
    def local_value_iteration(self, updated_cells: Set[Cell], max_iterations: int = 100,
                    convergence_tol: float = 1e-3, conservative_mode: bool = False, 
                    delta_max: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        LOCAL value iteration: update ALL leaves after refinement.
        """
        leaves = self.cell_tree.get_leaves()
        
        # ============ ADDED SECTION START ============
        # CRITICAL: Reinitialize ALL cells to l_lower/l_upper (required by Lemma 1)
        print(f"  Reinitializing ALL {len(leaves)} cells to l_lower/l_upper (Lemma 1 requirement)")
        reinit_start = time.time()

        # Use existing parallel initialization
        self.initialize_cells()  # This REINITIALIZE V to stage cost (Lemma 1 requirement)

        reinit_time = time.time() - reinit_start
        print(f"  ✓ Reinitialized {len(leaves)} cells in {reinit_time:.2f}s ({len(leaves)/reinit_time:.1f} cells/s)")
        # ============ ADDED SECTION END ============
            
            
        
        
        
        
        print(f"  Local VI: updating all {len(leaves)} cells")
        print(f"    Conservative mode: {conservative_mode}")
        if conservative_mode:
            print(f"    δ_max: {delta_max}")
        else:
            print(f"    Convergence tolerance: {convergence_tol}")
        print(f"    Max iterations: {max_iterations}")
        
        if len(leaves) == 0:
            print("    No cells to update!")
            return np.array([]), np.array([])
        
        # Update successor cache for new cells
        if updated_cells:
            print(f"    Updating successor cache for {len(updated_cells)} updated cells...")
            self._update_successor_cache_for_new_cells(updated_cells)
        
        # Use persistent pool
        if self.pool is None:
            self.pool = Pool(self.n_workers)
        
        conv_history_upper = []
        conv_history_lower = []
        
        for iteration in range(max_iterations):
            prev_upper = {cell.cell_id: cell.V_upper for cell in leaves}
            prev_lower = {cell.cell_id: cell.V_lower for cell in leaves}
            
            
            
            # Build numpy arrays for ALL leaves
            n_cells = len(leaves)
            cell_ids = np.array([c.cell_id for c in leaves])
            V_upper_arr = np.array([c.V_upper for c in leaves])
            V_lower_arr = np.array([c.V_lower for c in leaves])
            l_upper_arr = np.array([c.l_upper for c in leaves])
            l_lower_arr = np.array([c.l_lower for c in leaves])
            r_upper_arr = np.array([c.r_upper for c in leaves])  ##changed for RA case
            r_lower_arr = np.array([c.r_lower for c in leaves])  ##changed for RA case
            
            id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
            
            # Prepare tasks for ALL cells
            tasks = []
            for idx, cell in enumerate(leaves):
                succ_indices_by_action = []
                for action in self.env.get_action_space():
                    cache_key = (cell.cell_id, action)
                    if cache_key in self.successor_cache:
                        succ_ids = self.successor_cache[cache_key]
                        succ_indices = [id_to_idx[sid] for sid in succ_ids if sid in id_to_idx]
                    else:
                        succ_indices = []
                    succ_indices_by_action.append(succ_indices)
                tasks.append((idx, succ_indices_by_action))
            
            shared_data = (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, r_upper_arr, r_lower_arr, self.gamma)  ##changed for RA case
            
            worker_func = partial(_bellman_update_optimized, shared_data=shared_data)
            results = self.pool.map(worker_func, tasks)
            
            # Apply updates
            for cell_idx, new_upper, new_lower in results:
                leaves[cell_idx].V_upper = new_upper
                leaves[cell_idx].V_lower = new_lower
            
            # Check convergence
            diff_upper = max(abs(leaves[i].V_upper - prev_upper[leaves[i].cell_id]) for i in range(n_cells))
            diff_lower = max(abs(leaves[i].V_lower - prev_lower[leaves[i].cell_id]) for i in range(n_cells))
            
            # Store convergence history
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            if conservative_mode:
                # Conservative stopping condition
                delta_k = min(leaves[i].V_lower - prev_lower[leaves[i].cell_id] for i in range(n_cells))
                
                # Local VI convergence metrics
                print(f"    Local Iteration {iteration + 1:2d}: "
                    f"||V̄^k - V̄^k-1||_∞ = {diff_upper:10.30f}, "
                    f"||V_^k - V_^k-1||_∞ = {diff_lower:10.30f}, "
                    f"δ^k = {delta_k:12.20f}")
                
                # MODIFIED: Check both V_upper convergence AND conservative condition for V_lower
                if delta_k >= -delta_max and diff_upper < convergence_tol:
                    print(f"\n" + "="*60)
                    print("✓ CONSERVATIVE STOPPING CONDITION MET (Algorithm 3)")
                    print("="*60)
                    print(f"  δ^k = {delta_k:.10e} ≥ -δ_max = -{delta_max:.10e}")
                    print(f"  ||V̄^k - V̄^k-1||_∞ = {diff_upper:.10e} < {convergence_tol}")
                    
                    delta_to_use = abs(delta_k)
                    epsilon_cons = (self.gamma * delta_to_use) / (1 - self.gamma)
                    
                    print(f"  ε_cons = ({self.gamma} * {delta_to_use:.10e}) / {1 - self.gamma} = {epsilon_cons:.10e}")
                    print(f"  Correcting V_lower for {len(leaves)} cells: V_lower ← V_lower - ε_cons - ε_machine")
                    
                    # NEW: Track statistics
                    eps_machine = np.finfo(float).eps
                    cells_near_zero = 0
                    cells_flipped_sign = 0
                    
                    for cell in leaves:
                        old_value = cell.V_lower
                        
                        # Apply conservative correction
                        cell.V_lower = cell.V_lower - epsilon_cons - eps_machine
                        
                        # Track cells affected by cancellation
                        if abs(old_value - epsilon_cons) < 100 * eps_machine:
                            cells_near_zero += 1
                        
                        # Track sign flips due to eps_machine subtraction
                        if old_value - epsilon_cons > 0 and cell.V_lower <= 0:
                            cells_flipped_sign += 1
                    
                    print(f"  ✓ Conservative correction applied successfully")
                    print(f"  Machine epsilon ε_machine = {eps_machine:.3e}")
                    
                    if cells_near_zero > 0:
                        print(f"  ⚠️  {cells_near_zero} cells experienced near-cancellation")
                    if cells_flipped_sign > 0:
                        print(f"  ⚠️  {cells_flipped_sign} cells changed from (+) to (≤0) due to ε_machine")
                    break
            else:
                # Standard local convergence metrics
                print(f"    Local Iteration {iteration + 1:2d}: "
                    f"||V̄^k - V̄^k-1||_∞ = {diff_upper:10.20f}, "
                    f"||V_^k - V_^k-1||_∞ = {diff_lower:10.20f}")
                
                if diff_upper < convergence_tol and diff_lower < convergence_tol:
                    print(f"    " + "="*30)
                    print(f"    ✓ LOCAL CONVERGENCE ACHIEVED")
                    print(f"    " + "="*30)
                    print(f"      ||V̄^k - V̄^k-1||_∞ = {diff_upper:.20e} < {convergence_tol}")
                    print(f"      ||V_^k - V_^k-1||_∞ = {diff_lower:.20e} < {convergence_tol}")
                    print(f"      Converged in {iteration + 1} local iterations")
                    break
                    # Distribution of successor counts
            from collections import Counter
            successor_counts = [len(succ_ids) for succ_ids in self.successor_cache.values()]
            count_dist = Counter(successor_counts)

                # ============ ADD THIS SECTION START ============
        # # Compute average successors per cell
        # print(f"\n{'='*80}")
        # print("SUCCESSOR STATISTICS")
        # print(f"{'='*80}")

        # # Group successors by cell
        # from collections import defaultdict
        # successors_by_cell = defaultdict(list)
        # for (cell_id, action), successor_ids in self.successor_cache.items():
        #     successors_by_cell[cell_id].append(len(successor_ids))

        # # Compute statistics
        # actions = self.env.get_action_space()
        # n_cells = len(leaves)
        # n_actions = len(actions)
        # total_pairs = len(self.successor_cache)

        # # Per-cell averages
        # avg_successors_per_cell = []
        # for cell_id in range(n_cells):
        #     if cell_id in successors_by_cell:
        #         avg_for_this_cell = np.mean(successors_by_cell[cell_id])
        #         avg_successors_per_cell.append(avg_for_this_cell)
        #     else:
        #         avg_successors_per_cell.append(0.0)

        # # Overall statistics
        # overall_avg = np.mean([len(succ_ids) for succ_ids in self.successor_cache.values()])
        # overall_min = np.min([len(succ_ids) for succ_ids in self.successor_cache.values()])
        # overall_max = np.max([len(succ_ids) for succ_ids in self.successor_cache.values()])
        # overall_std = np.std([len(succ_ids) for succ_ids in self.successor_cache.values()])

        # print(f"Total (cell, action) pairs: {total_pairs} ({n_cells} cells × {n_actions} actions)")
        # print(f"\nSuccessors per (cell, action) pair:")
        # print(f"  Mean: {overall_avg:.2f}")
        # print(f"  Std:  {overall_std:.2f}")
        # print(f"  Min:  {overall_min}")
        # print(f"  Max:  {overall_max}")

        # # Per-cell statistics
        # cell_avg_mean = np.mean(avg_successors_per_cell)
        # cell_avg_std = np.std(avg_successors_per_cell)
        # cell_avg_min = np.min(avg_successors_per_cell)
        # cell_avg_max = np.max(avg_successors_per_cell)

        # print(f"\nAverage successors per cell (averaged over actions):")
        # print(f"  Mean: {cell_avg_mean:.2f}")
        # print(f"  Std:  {cell_avg_std:.2f}")
        # print(f"  Min:  {cell_avg_min:.2f}")
        # print(f"  Max:  {cell_avg_max:.2f}")

        # # Distribution of successor counts
        # from collections import Counter
        # successor_counts = [len(succ_ids) for succ_ids in self.successor_cache.values()]
        # count_dist = Counter(successor_counts)

        # print(f"\nDistribution of successor counts:")
        # for count in sorted(count_dist.keys()):
        #     freq = count_dist[count]
        #     pct = 100 * freq / total_pairs
        #     bar = '█' * int(pct / 2)  # Simple bar chart
        #     print(f"  {count:3d} successors: {freq:4d} pairs ({pct:5.1f}%) {bar}")

        # # Cells with zero successors
        # zero_successor_pairs = [(cell_id, action) for (cell_id, action), succ_ids in self.successor_cache.items() if len(succ_ids) == 0]
        # cells_with_zero = set(cell_id for cell_id, action in zero_successor_pairs)

        # if len(zero_successor_pairs) > 0:
        #     print(f"\n⚠️  WARNING: {len(zero_successor_pairs)} (cell, action) pairs have ZERO successors!")
        #     print(f"   {len(cells_with_zero)}/{n_cells} cells have at least one action with no successors")
        # else:
        #     print(f"\n✓ All (cell, action) pairs have at least one successor")

        # print(f"{'='*80}\n")
        # # ============ ADD THIS SECTION END ============

        # Final local iteration summary
        if iteration == max_iterations - 1:
            print(f"    " + "!"*30)
            print(f"    ⚠️  LOCAL VI REACHED MAX ITERATIONS ({max_iterations})")
            print(f"    " + "!"*30)
            if conservative_mode:
                delta_k = min(leaves[i].V_lower - prev_lower[leaves[i].cell_id] for i in range(n_cells))
                print(f"      Final δ^k = {delta_k:.20e}")
            print(f"      Final ||V̄^k - V̄^k-1||_∞ = {diff_upper:.20e}")
            print(f"      Final ||V_^k - V_^k-1||_∞ = {diff_lower:.20e}")
            
            # Apply conservative correction even if max iterations reached
            if conservative_mode:
                # MODIFIED: Use min(|delta_k|, delta_max) when max iterations reached too
                delta_to_use = abs(delta_k)#delta_to_use = min(abs(delta_k), delta_max) #if delta_k < 0 else delta_max
                epsilon_cons = (self.gamma * delta_to_use) / (1 - self.gamma)
                
                # print(f"      Using δ_actual = {delta_to_use:.10e} for correction")
                print(f"      Applying conservative margin: ε_cons = {epsilon_cons:.10e}")
                for cell in leaves:
                    cell.V_lower = cell.V_lower - epsilon_cons
                print(f"      ✓ Conservative correction applied")
        
    
        print(f"    Local VI completed in {iteration + 1} iterations")
        
        # Print final convergence values
        if len(conv_history_upper) > 0:
            final_upper = conv_history_upper[-1]
            final_lower = conv_history_lower[-1]
            print(f"    Final convergence values:")
            print(f"      ||V̄^final - V̄^prev||_∞ = {final_upper:.8e}")
            print(f"      ||V_^final - V_^prev||_∞ = {final_lower:.8e}")
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
    def _update_successor_cache_for_new_cells(self, new_cells: Set[Cell]):
        """Update successor cache using HYBRID approach (parallel ODE + spatial index)."""
        if not new_cells:
            return
        
        print(f"    Updating successor cache for {len(new_cells)} new cells (HYBRID)...")
        start_time = time.time()
        
        actions = self.env.get_action_space()
        
        # Identify refined parent cells
        refined_parent_ids = set()
        for cell in new_cells:
            if cell.parent is not None:
                refined_parent_ids.add(cell.parent.cell_id)
        
        # Remove cache for refined parents AND cells pointing to them
        keys_to_delete = []
        for key in self.successor_cache.keys():
            cell_id, action = key
            successor_ids = self.successor_cache[key]
            
            if cell_id in refined_parent_ids or any(sid in refined_parent_ids for sid in successor_ids):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.successor_cache[key]
        
        # Find all affected cells
        affected_cells = set(new_cells)
        all_leaves = self.cell_tree.get_leaves()
        for cell in all_leaves:
            for action in actions:
                key = (cell.cell_id, action)
                if key not in self.successor_cache:
                    affected_cells.add(cell)
                    break
        
        print(f"      Affected cells: {len(affected_cells)}")
        
        # STEP 1: Parallel ODE integration
        ode_tasks = []
        for cell in affected_cells:
            for action in actions:
                ode_tasks.append((
                    cell.cell_id,
                    cell.center.copy(),
                    action,
                    self.env,
                    self.reachability.tau,
                    self.reachability.dt
                ))
        
        chunksize = max(1, len(ode_tasks) // (self.n_workers * 4))
        if self.pool is not None:
            trajectory_results = self.pool.map(_compute_checkpoint_worker, ode_tasks, chunksize=chunksize)
        else:
            with Pool(self.n_workers) as pool:
                trajectory_results = pool.map(_compute_checkpoint_worker, ode_tasks, chunksize=chunksize)
        
        # STEP 2: Sequential successor queries using spatial index
        cell_dict = {cell.cell_id: cell for cell in all_leaves}
        
        for cell_id, action, checkpoint_states in trajectory_results:
            cell = cell_dict[cell_id]
            
            # Compute cell radius
            if self.reachability.use_infinity_norm:
                r = 0.5 * cell.get_max_range()
            else:
                ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
                r = 0.5 * np.linalg.norm(ranges)
            
            successor_cell_ids = set()
            
            for i, state_at_checkpoint in enumerate(checkpoint_states):
                growth_factor_i = self.reachability.growth_factors[i]
                expansion_i = r * growth_factor_i
                
                reach_bounds = np.zeros((self.env.get_state_dim(), 2))
                reach_bounds[0, 0] = state_at_checkpoint[0] - expansion_i
                reach_bounds[0, 1] = state_at_checkpoint[0] + expansion_i
                reach_bounds[1, 0] = state_at_checkpoint[1] - expansion_i
                reach_bounds[1, 1] = state_at_checkpoint[1] + expansion_i
                
                if self.env.get_state_dim() == 3:
                    theta_lower = state_at_checkpoint[2] - expansion_i
                    theta_upper = state_at_checkpoint[2] + expansion_i
                    # theta_lower, theta_upper = self.reachability.fix_angle_interval(theta_lower, theta_upper)
                    reach_bounds[2, 0] = theta_lower
                    reach_bounds[2, 1] = theta_upper
                
                # USE SPATIAL INDEX
                candidates = self.cell_tree.get_intersecting_cells(reach_bounds)
                
                for candidate in candidates:
                    # if self.reachability._check_theta_intersection(candidate, reach_bounds):
                    successor_cell_ids.add(candidate.cell_id)
                        
            # # ##changed for RA case: If no successors found, add self-loop. checking if solving large values of V issue when no successors and it fixes value to \overline or \underline V
            # if len(successor_cell_ids) == 0:  ##changed for RA case
            #     successor_cell_ids.add(cell_id)  ##changed for RA case
            #     print(f"      Cell {cell_id} with action {action}: No successors, adding self-loop")  ##changed for RA case
        
            key = (cell_id, action)
            self.successor_cache[key] = list(successor_cell_ids)
        
        elapsed = time.time() - start_time
        print(f"    ✓ Cache updated in {elapsed:.2f}s ({len(ode_tasks)/elapsed:.1f} tasks/s)")
    
    def _identify_boundary_cells(self) -> List[Cell]:
        """Identify boundary cells where V̄_γ(s) > 0 and V_γ(s) < 0."""
        boundary = []
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                # if cell.V_upper > 0 and cell.V_lower < 0:
                if cell.V_upper > 0 and cell.V_lower <= 0:#THIS IS THE CORRECT ONE FIX LATER
                    boundary.append(cell)
        return boundary
    
    def _save_plot(self, iteration: int, final: bool = False):
        """Save visualization with refinement phase in filename."""
        suffix = "_final" if final else ""
        
        if self.refinement_phase > 0:
            filename = os.path.join(
                self.output_dir, 
                f"iteration_{iteration:04d}_refinement_{self.refinement_phase:02d}{suffix}.png"
            )
        else:
            filename = os.path.join(
                self.output_dir, 
                f"iteration_{iteration:04d}{suffix}.png"
            )
        
        plot_value_function(self.env, self.cell_tree, filename, iteration)


# ============================================================================
# PART 6: ADAPTIVE REFINEMENT (ALGORITHM 2)
# ============================================================================

class AdaptiveRefinement:
    """Algorithm 2 with optimized parallelization and spatial indexing."""
    
    def __init__(self, args, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: GronwallReachabilityAnalyzer, output_dir: str = None):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.args = args
        
        if output_dir is None:
            rname = type(reachability).__name__
            param_suffix = (
                f"dynamics_{args.dynamics}_"
                f"gamma_{args.gamma:.3f}_"
                f"dt_{env.dt:.3f}_"
                f"tau_{env.tau:.3f}_"
                f"tol_{args.tolerance:.1e}_"
                f"eps_{args.epsilon:.3f}"
                f"vi-iterations_{args.vi_iterations}"
                f"conservative_{args.conservative}"
                f"delta-max_{args.delta_max}"
                f"init_resol_{args.initial_resolution}"
            )
            output_dir = os.path.join(
                "./results_RA/fixed_bellmannew/machineeps",
                f"{rname}_{param_suffix}"
            )
        self.output_dir = output_dir
        
        self.value_iterator = SafetyValueIterator(
            env, gamma, cell_tree, reachability, output_dir,
            n_workers=args.workers, precompute_successors=args.precompute,args=args
        )
        self.L_l = env.get_lipschitz_constants()[1]
    
    def refine(self, epsilon: float, max_refinements: int = 100,
            vi_iterations_per_refinement: int = 100):
        """Main adaptive refinement loop (Algorithm 2)."""
        eta_min = epsilon / (2 * self.L_l)  # From paper: η_min = ε/(2L_l)
        
        print(f"\n" + "="*70)
        print("ADAPTIVE REFINEMENT CONFIGURATION")
        print("="*70)
        print(f"  Error tolerance ε: {epsilon}")
        print(f"  Minimum cell size η_min: {eta_min:.6f}")
        print(f"  Maximum refinements: {max_refinements}")
        print(f"  VI iterations per refinement: {vi_iterations_per_refinement}")
        print(f"  Conservative mode: {self.args.conservative}")
        if self.args.conservative:
            print(f"  δ_max: {self.args.delta_max}")
            epsilon_cons = (self.args.gamma * self.args.delta_max) / (1 - self.args.gamma)
            print(f"  Conservative margin ε_cons: {epsilon_cons:.10e}")
        
        # Phase 0: Initial value iteration on full grid
        print(f"\n" + "="*70)
        print("PHASE 0: INITIAL VALUE ITERATION")
        print("="*70)
        print(f"Grid: {self.cell_tree.get_num_leaves()} cells")
        
        self.value_iterator.refinement_phase = 0
        start_timee = time.time() 
        conv_upper, conv_lower = self.value_iterator.value_iteration(
            max_iterations=vi_iterations_per_refinement,
            plot_freq=self.args.plot_freq,
            conservative_mode=self.args.conservative,
            delta_max=self.args.delta_max
        )
        elapsedd = time.time() - start_timee
        print(f"total time for running value_iteration the first time (phase 0) on the initial grid including grid initialization, setting up cpu parallelism, successor set computation and value iteration is {elapsedd} seconds")
        # Print initial convergence summary
        if len(conv_upper) > 0:
            print(f"Initial VI completed in {len(conv_upper)} iterations")
            print(f"Final convergence: ||V̄||_∞ = {conv_upper[-1]:.8e}, ||V_||_∞ = {conv_lower[-1]:.8e}")
        
        # Save plot after initial VI
        filename = os.path.join(self.output_dir, "value_function_phase_0_complete.png")
        # plot_value_function(self.env, self.cell_tree, filename, 0)
        print(f"Saved initial visualization: {filename}")
        
        # Initial queue state
        boundary_cells = self._identify_boundary_cells()
        refinable = [c for c in boundary_cells if c.get_max_range() > eta_min]
        too_small = [c for c in boundary_cells if c.get_max_range() <= eta_min]
        
        print(f"\n" + "="*70)
        print("INITIAL QUEUE STATE")
        print("="*70)
        print(f"Total boundary cells: {len(boundary_cells)}")
        print(f"  Refinable (>η_min={eta_min:.6f}): {len(refinable)}")
        print(f"  Below threshold: {len(too_small)}")
        
        # Refinement loop (Algorithm 2)
        refinement_iter = 0
        total_refined = 0
        
        while refinement_iter < max_refinements and len(refinable) > 0:
            boundary_cells = self._identify_boundary_cells()
            refinable = [c for c in boundary_cells if c.get_max_range() > eta_min]
            too_small = [c for c in boundary_cells if c.get_max_range() <= eta_min]
            
            print(f"\n" + "="*70)
            print(f"REFINEMENT PHASE {refinement_iter + 1}")
            print("="*70)
            phase_start = time.time()
            print(f"Boundary cells: {len(boundary_cells)}")
            print(f"  Refinable (>η_min={eta_min:.6f}): {len(refinable)}")
            print(f"  Below threshold: {len(too_small)}")
            print(f"  Cumulative refined: {total_refined}")
            
            if len(refinable) == 0:
                print("  No refinable cells remaining - stopping refinement")
                break
            
            # Cell size statistics
            if refinable:
                sizes = [c.get_max_range() for c in refinable]
                print(f"  Refinable cell size statistics:")
                print(f"    Max:    {max(sizes):.6f}")
                print(f"    Mean:   {np.mean(sizes):.6f}")
                print(f"    Min:    {min(sizes):.6f}")
                print(f"    Median: {np.median(sizes):.6f}")
            
            # Perform refinement (Algorithm 2, lines 8-13)
            print(f"  Refining {len(refinable)} cells...")
            new_cells = []
            refinement_start = time.time()
            
            for cell in refinable:
                self.cell_tree.refine_cell(cell)
                new_cells.extend(cell.children)

            refinement_time = time.time() - refinement_start
            total_refined += len(refinable)
            
            print(f"    Refined: {len(refinable)} parent cells")
            print(f"    Created: {len(new_cells)} child cells")
            print(f"    Total cells: {self.cell_tree.get_num_leaves()}")
            print(f"    Refinement time: {refinement_time:.3f}s")
            print(f"    Cumulative refined: {total_refined}")
            
            # Rebuild spatial index after refinements
            print(f"  Rebuilding spatial index...")
            
            print(f"    Pre-rebuild: {len(self.cell_tree.leaves)} leaves")
            non_leaves_before = [c for c in self.cell_tree.leaves if not c.is_leaf]
            print(f"    Non-leaves before rebuild: {len(non_leaves_before)}")

            
            index_start = time.time()
            self.cell_tree.rebuild_spatial_index()
            index_time = time.time() - index_start
            print(f"    Spatial index rebuilt in {index_time:.3f}s")
            print(f"    Post-rebuild: {len(self.cell_tree.leaves)} leaves")
            non_leaves_after = [c for c in self.cell_tree.leaves if not c.is_leaf]
            print(f"    Non-leaves after rebuild: {len(non_leaves_after)}")

            if len(non_leaves_after) > len(non_leaves_before):
                print(f"    ❌ ERROR: Rebuild ADDED {len(non_leaves_after) - len(non_leaves_before)} non-leaf cells!")
                        
            # Initialize new cells
            if new_cells:
                print(f"  Initializing {len(new_cells)} new cells...")
                init_start = time.time()
                self.value_iterator.initialize_new_cells(new_cells)
                init_time = time.time() - init_start
                print(f"    Initialization completed in {init_time:.3f}s")
            
            # Set refinement phase
            self.value_iterator.refinement_phase = refinement_iter + 1
            
            # LOCAL value iteration (Algorithm 2, line 18)
            print(f"  Starting local value iteration...")
            local_vi_start = time.time()
            
            conv_upper, conv_lower = self.value_iterator.local_value_iteration(
                updated_cells=set(new_cells),
                max_iterations=vi_iterations_per_refinement,
                convergence_tol=self.args.tolerance,
                conservative_mode=self.args.conservative,
                delta_max=self.args.delta_max
            )
            
            local_vi_time = time.time() - local_vi_start
                # >>> ADD THIS SECTION HERE <
            ## Local VI convergence summary
            # print(f"  Local VI completed in {len(conv_upper)} iterations, time: {local_vi_time:.3f}s")
            

            # print(f"\n  Cell classification after Phase {refinement_iter + 1}:")
            # safe = unsafe = boundary = 0
            # for cell in self.cell_tree.get_leaves():
            #     if cell.V_lower is not None and cell.V_upper is not None:
            #         if cell.V_lower > 0:
            #             safe += 1
            #         elif cell.V_upper <= 0:
            #             unsafe += 1
            #         else:
            #             boundary += 1

            # total = self.cell_tree.get_num_leaves()
            # print(f"    Safe:     {safe:6d} ({100*safe/total:5.1f}%)")
            # print(f"    Unsafe:   {unsafe:6d} ({100*unsafe/total:5.1f}%)")
            # print(f"    Boundary: {boundary:6d} ({100*boundary/total:5.1f}%)")
            # print(f"    Total:    {total:6d}")
            # >>> END SECTION <
            # Local VI convergence summary
            
            print(f"  Local VI completed in {len(conv_upper)} iterations, time: {local_vi_time:.3f}s")
            
            # >>> INSERT PROBE LOGGING HERE <<<
            #>>> INSERT PROBE LOGGING HERE <
# >>> INSERT PROBE LOGGING HERE <
# >>> INSERT PROBE LOGGING HERE <
            # x_star = np.array([-3, 3, 3.14])
            # cell, vl, vu = self.probe_state_value(self.cell_tree, x_star)
            # if cell is None:
            #     print(f"[Phase {refinement_iter+1}] x* is not contained in any leaf cell!?")
            # else:
            #     print(f"[Phase {refinement_iter+1}] Probing x* = {x_star}:")
            #     print(f"  Cell ID: {cell.cell_id}")
            #     print(f"  Center: {cell.center}")
            #     print(f"  Bounds: x:({cell.bounds[0][0]:.20f},{cell.bounds[0][1]:.20f}), "
            #         f"y:({cell.bounds[1][0]:.20f},{cell.bounds[1][1]:.20f}), "
            #         f"theta:({cell.bounds[2][0]:.20f},{cell.bounds[2][1]:.20f})")
            #     print(f"  V_lower = {vl:.30f}")
            #     print(f"  V_upper = {vu:.30f}")
            #     print(f"  l_lower = {cell.l_lower:.8f}")
            #     print(f"  l_upper = {cell.l_upper:.8f}")
            #     print(f"  r_lower = {cell.r_lower:.8f}")
            #     print(f"  r_upper = {cell.r_upper:.8f}")
                
            #     # Compute cell radius
            #     if self.reachability.use_infinity_norm:
            #         r = 0.5 * cell.get_max_range()
            #     else:
            #         ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
            #         r = 0.5 * np.linalg.norm(ranges)
            #     print(f"  Cell radius: {r:.8f}")
                
            #     # Get all successors for this cell across all actions
            #     print(f"\n  Successors for Cell {cell.cell_id}:")
            #     actions = self.env.get_action_space()
            #     cell_id_to_cell = {c.cell_id: c for c in self.cell_tree.get_leaves()}
                
            #     for action in actions:
            #         cache_key = (cell.cell_id, action)
            #         if cache_key in self.value_iterator.successor_cache:
            #             succ_ids = self.value_iterator.successor_cache[cache_key]
            #             print(f"\n    Action {action:+.3f} → {len(succ_ids)} successors:")
                        
            #             # Compute trajectory for this action
            #             checkpoint_states = self.env.dynamics_multi_step(
            #                 cell.center, action, self.reachability.tau, self.reachability.dt
            #             )
                        
            #             # Show reachable bounds at each checkpoint
            #             print(f"      Reachable bounds at checkpoints:")
            #             for i, state_at_checkpoint in enumerate(checkpoint_states):
            #                 growth_factor_i = self.reachability.growth_factors[i]
            #                 expansion_i = r * growth_factor_i
                            
            #                 reach_bounds = np.zeros((self.env.get_state_dim(), 2))
            #                 reach_bounds[0, 0] = state_at_checkpoint[0] - expansion_i
            #                 reach_bounds[0, 1] = state_at_checkpoint[0] + expansion_i
            #                 reach_bounds[1, 0] = state_at_checkpoint[1] - expansion_i
            #                 reach_bounds[1, 1] = state_at_checkpoint[1] + expansion_i
                            
            #                 theta_lower = state_at_checkpoint[2] - expansion_i
            #                 theta_upper = state_at_checkpoint[2] + expansion_i
            #                 reach_bounds[2, 0] = theta_lower
            #                 reach_bounds[2, 1] = theta_upper
                            
            #                 t = self.reachability.checkpoint_times[i]
            #                 print(f"        t={t:.3f}s: state=({state_at_checkpoint[0]:+.6f}, {state_at_checkpoint[1]:+.6f}, {state_at_checkpoint[2]:+.6f})")
            #                 print(f"          expansion={expansion_i:.6f}, growth_factor={growth_factor_i:.6f}")
            #                 print(f"          x:[{reach_bounds[0,0]:+.6f}, {reach_bounds[0,1]:+.6f}]")
            #                 print(f"          y:[{reach_bounds[1,0]:+.6f}, {reach_bounds[1,1]:+.6f}]")
            #                 print(f"          θ:[{reach_bounds[2,0]:+.6f}, {reach_bounds[2,1]:+.6f}]")
                        
            #             if len(succ_ids) == 0:
            #                 print(f"      (No successors)")
            #             else:
            #                 print(f"\n      Successor cells:")
            #                 for succ_id in succ_ids:
            #                     if succ_id in cell_id_to_cell:
            #                         succ = cell_id_to_cell[succ_id]
            #                         print(f"        Cell {succ.cell_id}:")
            #                         print(f"          Bounds: x:({succ.bounds[0][0]:+.6f},{succ.bounds[0][1]:+.6f}), "
            #                               f"y:({succ.bounds[1][0]:+.6f},{succ.bounds[1][1]:+.6f}), "
            #                               f"theta:({succ.bounds[2][0]:+.6f},{succ.bounds[2][1]:+.6f})")
            #                         print(f"          V_lower = {succ.V_lower:.30f}, V_upper = {succ.V_upper:.30f}")
            #                         print(f"          l_lower = {succ.l_lower:.8f}, l_upper = {succ.l_upper:.8f}")
            #                         print(f"          r_lower = {succ.r_lower:.8f}, r_upper = {succ.r_upper:.8f}")
                                    
            #                         # NEW: Show successors of this successor (one more layer)
            #                         print(f"          Successors of Cell {succ.cell_id}:")
            #                         for action2 in actions:
            #                             cache_key2 = (succ.cell_id, action2)
            #                             if cache_key2 in self.value_iterator.successor_cache:
            #                                 succ_succ_ids = self.value_iterator.successor_cache[cache_key2]
            #                                 print(f"            Action {action2:+.3f} → {len(succ_succ_ids)} successors")
                                            
            #                                 if len(succ_succ_ids) > 0:
            #                                     # Show details of each successor's successor
            #                                     for succ_succ_id in succ_succ_ids[:3]:  # Limit to first 3 to avoid clutter
            #                                         if succ_succ_id in cell_id_to_cell:
            #                                             succ_succ = cell_id_to_cell[succ_succ_id]
            #                                             print(f"              → Cell {succ_succ.cell_id}:")
            #                                             print(f"                 Bounds: x:({succ_succ.bounds[0][0]:+.6f},{succ_succ.bounds[0][1]:+.6f}), "
            #                                                   f"y:({succ_succ.bounds[1][0]:+.6f},{succ_succ.bounds[1][1]:+.6f}), "
            #                                                   f"theta:({succ_succ.bounds[2][0]:+.6f},{succ_succ.bounds[2][1]:+.6f})")
            #                                             print(f"                 V_lower={succ_succ.V_lower:.30f}, V_upper={succ_succ.V_upper:.30f}")
            #                                             print(f"                 l_lower={succ_succ.l_lower:.6f}, r_lower={succ_succ.r_lower:.6f}")
                                                        
            #                                             # NEW: Count successors of this third-layer cell
            #                                             total_third_layer_succs = 0
            #                                             for action3 in actions:
            #                                                 cache_key3 = (succ_succ.cell_id, action3)
            #                                                 if cache_key3 in self.value_iterator.successor_cache:
            #                                                     total_third_layer_succs += len(self.value_iterator.successor_cache[cache_key3])
            #                                             print(f"                 Total successors (all actions): {total_third_layer_succs}")
                                                        
            #                                     if len(succ_succ_ids) > 3:
            #                                         print(f"              ... and {len(succ_succ_ids) - 3} more")
            #                             else:
            #                                 print(f"            Action {action2:+.3f}: (no cache entry)")
                                    
            #                     else:
            #                         print(f"        Cell {succ_id}: (not found in current leaves)")
            #         else:
            #             print(f"\n    Action {action:+.3f}: (no cache entry)")
                
                # print("-"*60)
            # >>> END INSERT <
            # >>> END INSERT <
            # >>> END INSERT <
            # >>> END INSERT <<<
            if len(conv_upper) > 0:
                final_upper = conv_upper[-1] if len(conv_upper) > 0 else float('inf')
                final_lower = conv_lower[-1] if len(conv_lower) > 0 else float('inf')
                print(f"    Final ||V̄^k - V̄^k-1||_∞ = {final_upper:.20e}")
                print(f"    Final ||V_^k - V_^k-1||_∞ = {final_lower:.20e}")
            
            # Save plot
            filename = os.path.join(
                self.output_dir, 
                f"value_function_phase_{refinement_iter + 1}_complete.png"
            )

            plot_value_function(self.env, self.cell_tree, filename, refinement_iter + 1)
            print(f"  Saved visualization: {filename}")
            
            # Phase completion summary
            phase_time = time.time() - phase_start  # Use phase_start instead
            phase_num_leaves = self.cell_tree.get_num_leaves()
            # print(f"  Phase {refinement_iter + 1} completed in {phase_time:.3f}s")
            print(f"  Phase {refinement_iter + 1} completed in {phase_time:.2f}s with {phase_num_leaves} leaf cells")
            
            refinement_iter += 1
        
        # Final summary
        print(f"\n" + "="*70)
        print("ADAPTIVE REFINEMENT COMPLETE")
        print("="*70)
        print(f"Refinement Summary:")
        print(f"  Phases completed: {refinement_iter}")
        print(f"  Parent cells refined: {total_refined}")
        print(f"  Child cells created: {total_refined * 2}")
        print(f"  Final cells: {self.cell_tree.get_num_leaves()}")
        
        # Final boundary state
        final_boundary = self._identify_boundary_cells()
        final_refinable = [c for c in final_boundary if c.get_max_range() > eta_min]
        print(f"Final Boundary State:")
        print(f"  Boundary cells: {len(final_boundary)}")
        print(f"  Still refinable: {len(final_refinable)}")
        print(f"  Below threshold: {len(final_boundary) - len(final_refinable)}")
        
        self._print_statistics()
        self.reachability.print_debug_summary()
        print(f"\n✓ All results saved to: {self.output_dir}/")
        
    def probe_state_value(self,cell_tree, x_star):
        for cell in cell_tree.get_leaves():
            if cell.contains_point(x_star):
                return cell, cell.V_lower, cell.V_upper
        return None, None, None

    def _identify_boundary_cells(self) -> List[Cell]:
        """Identify boundary cells where V̄_γ(s) > 0 and V_γ(s) < 0."""
        # CHANGED: Now uses V_lower directly (which may contain conservative correction)
        return self.value_iterator._identify_boundary_cells()
    
    def _print_statistics(self):
        """Print final cell classification statistics."""
        safe = unsafe = boundary = 0
        for cell in self.cell_tree.get_leaves():
            if cell.V_lower is not None and cell.V_upper is not None:
                if cell.V_lower > 0:
                    safe += 1
                elif cell.V_upper < 0:
                    unsafe += 1
                else:
                    boundary += 1
        
        total = self.cell_tree.get_num_leaves()
        print(f"\nFinal Cell Classification:")
        print(f"  Safe:     {safe:6d} ({100*safe/total:5.1f}%)")
        print(f"  Unsafe:   {unsafe:6d} ({100*unsafe/total:5.1f}%)")
        print(f"  Boundary: {boundary:6d} ({100*boundary/total:5.1f}%)")


# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

def plot_value_function(
    env: Environment,
    cell_tree: CellTree,
    filename: str,
    iteration: int,
    theta_slices: list = None
):
    """Plots 2D slices of the 3D value function for Dubins car."""
    if theta_slices is None:
        # theta_slices = [0, np.pi/4, np.pi/2, np.pi]
        theta_slices = [0,np.pi,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2]
    
    n_slices = len(theta_slices)
    fig, axes = plt.subplots(3, n_slices, figsize=(5*n_slices, 14),dpi=600)
    
    if n_slices == 1:
        axes = axes.reshape(3, 1)
    
    for idx, theta in enumerate(theta_slices):
        # Upper bound
        ax_upper = axes[0, idx]
        _plot_slice(env, cell_tree, theta, ax_upper, "V̄_γ", upper=True)
        ax_upper.set_title(f"Upper Bound V̄_γ (θ={theta:.2f} rad)")
        
        # Lower bound (CHANGED: Now uses V_lower which may contain conservative correction)
        ax_lower = axes[1, idx]
        _plot_slice(env, cell_tree, theta, ax_lower, "V_γ", upper=False)
        ax_lower.set_title(f"Lower Bound V_γ (θ={theta:.2f} rad)")
        
        # Classification
        ax_cls = axes[2, idx]
        _plot_classification_slice(env, cell_tree, theta, ax_cls)
        ax_cls.set_title(f"Cell Classification (θ={theta:.2f} rad)")
    
    fig.suptitle(f"Safety Value Function - Iteration {iteration}", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=800, bbox_inches='tight')
    plt.close()


def _plot_slice(
    env: Environment,
    cell_tree: CellTree,
    theta: float,
    ax,
    label: str,
    upper: bool
):
    """Plot value function by directly coloring leaf cells with values displayed."""

    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Collect values from all relevant cells to determine color scale
    values = []
    relevant_cells = []
    # cells_for_this_theta = []
    for cell in cell_tree.get_leaves():

        # Only consider cells whose theta range includes this slice
        theta_min, theta_max = cell.bounds[2]

        if not (theta_min <= theta <= theta_max):
            continue

        if upper and cell.V_upper is not None:
            values.append(cell.V_upper)
            relevant_cells.append((cell, cell.V_upper))

        elif not upper and cell.V_lower is not None:
            values.append(cell.V_lower)
            relevant_cells.append((cell, cell.V_lower))

    if not values:
        ax.text(0.5, 0.5, 'No data for this slice', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        return
    
    # Determine color scale
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    cmap = plt.cm.RdYlGn

    vmin, vmax = np.min(values), np.max(values)

    # If all values are the same, expand slightly
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1e-12, vmax + 1e-12

    # If all values are negative, still use full colormap range (no white)
    if vmax <= 0:
        norm = Normalize(vmin=vmin, vmax=0)
    elif vmin >= 0:
        norm = Normalize(vmin=0, vmax=vmax)
    else:
        # mixed sign values
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot each cell as a colored rectangle with value text
    for cell, value in relevant_cells:
        a_x, b_x = cell.bounds[0]
        a_y, b_y = cell.bounds[1]
        color = cmap(norm(value))
        
        # Draw rectangle
        rect = Rectangle((a_x, a_y), b_x - a_x, b_y - a_y,
                         facecolor=color, edgecolor='black', linewidth=0.005,
                         antialiased=False, alpha=1.0)
        ax.add_patch(rect)
        
        # Add text label with value at cell center
        cell_width = b_x - a_x
        cell_height = b_y - a_y
        
        # Only add text if cell is large enough
        min_cell_size = 0.05
        if cell_width > min_cell_size and cell_height > min_cell_size:
            cx = (a_x + b_x) / 2
            cy = (a_y + b_y) / 2
            
            # Format value in scientific notation (short format)
            text = f'{value:.1e}'
            
            # Choose text color based on background brightness
            normalized_value = norm(value)
            text_color = 'black' if normalized_value > 0.5 else 'white'
            
            # Determine font size to fit within cell
            fontsize = min(cell_width * 15, cell_height * 15, 6)
            
            ax.text(cx, cy, text, 
                   ha='center', va='center',
                   fontsize=fontsize, 
                   color=text_color,
                   weight='normal',
                   clip_on=True)

    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          facecolor='none', edgecolor='darkblue', linewidth=2, zorder=10)
        ax.add_patch(obstacle)
        
        # Draw target set  ##changed for RA case
        target = Circle(env.target_position, env.target_radius,  ##changed for RA case
                       facecolor='none', edgecolor='orange', linewidth=2, zorder=10, linestyle='--')  ##changed for RA case
        ax.add_patch(target)  ##changed for RA case
        
    elif isinstance(env, EvasionEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          facecolor='none', edgecolor='darkblue', linewidth=2, zorder=10)
        ax.add_patch(obstacle)

        # Draw target set  ##changed for RA case
        target = Circle(env.target_position, env.target_radius,  ##changed for RA case
                       facecolor='none', edgecolor='orange', linewidth=2, zorder=10, linestyle='--')  ##changed for RA case
        ax.add_patch(target)  ##changed for RA case
        

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)

    from matplotlib.ticker import FuncFormatter

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=label)

    # Use scientific notation
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    def safe_ticks(vmin, vmax, zero_threshold=0.1):
        """Plot vmin, vmax, and optionally 0 if both are sufficiently far from 0."""
        ticks = [vmin, vmax]
        if abs(vmin) >= zero_threshold and abs(vmax) >= zero_threshold:
            ticks.insert(1, 0.0)
        ticks = sorted(set(ticks))
        return ticks

    tick_vals = safe_ticks(vmin, vmax)
    cbar.set_ticks(tick_vals)
    cbar.ax.set_yticklabels([f'{v:.1e}' for v in tick_vals])
def _plot_classification_slice(
    env: Environment,
    cell_tree: CellTree,
    theta: float,
    ax
):
    """Plot cell classification by directly coloring leaf cells."""
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Colors for classification
    C_UNSAFE = "#d62728"
    C_SAFE = "#2ca02c"
    C_BOUND = "#7f7f7f"
    C_UNKNOWN = "#ffffff"
    
    # Plot each cell directly
    for cell in cell_tree.get_leaves():
        # Only plot cells whose theta range includes this slice
        theta_min, theta_max = cell.bounds[2]
        if not (theta_min <= theta <= theta_max):
            continue
        
        # Determine cell color based on classification
        # CHANGED: Uses V_lower directly (may contain conservative correction)
        if cell.V_upper is None or cell.V_lower is None:
            color = C_UNKNOWN
        elif cell.V_lower > 0:
            color = C_SAFE
        elif cell.V_upper <= 0:##changed this##changed this##changed this
            color = C_UNSAFE
        else:
            color = C_BOUND
        

        # Draw rectangle directly on the provided axes
        rect = Rectangle(
            (cell.bounds[0, 0], cell.bounds[1, 0]),
            cell.get_range(0), cell.get_range(1),
            facecolor=color,
            edgecolor='black',     # crisp outline
            linewidth=0.005,         # fine lines for dense grids
            antialiased=False,
            alpha=1.0
        )
        ax.add_patch(rect)

    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          facecolor='none', edgecolor='darkblue', linewidth=2, zorder=10)
        ax.add_patch(obstacle)
        
        # Draw target set  ##changed for RA case
        target = Circle(env.target_position, env.target_radius,  ##changed for RA case
                       facecolor='none', edgecolor='orange', linewidth=2, zorder=10, linestyle='--')  ##changed for RA case
        ax.add_patch(target)  ##changed for RA case
        
    elif isinstance(env, EvasionEnvironment):
        obstacle = Circle((0, 0), env.obstacle_radius,
                          facecolor='none', edgecolor='darkblue', linewidth=2, zorder=10)
        ax.add_patch(obstacle)
        # Draw target set  ##changed for RA case
        target = Circle(env.target_position, env.target_radius,  ##changed for RA case
                       facecolor='none', edgecolor='orange', linewidth=2, zorder=10, linestyle='--')  ##changed for RA case
        ax.add_patch(target)  ##changed for RA case
        
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)
    # Legend
    # legend_elems = [
    #     Patch(facecolor=C_SAFE, edgecolor='black', label='Safe (V_>0)'),
    #     Patch(facecolor=C_UNSAFE, edgecolor='black', label='Unsafe (V̄<0)'),
    #     Patch(facecolor=C_BOUND, edgecolor='black', label='Boundary (mixed)')
    # ]
    # ax.legend(handles=legend_elems, loc='upper right', fontsize=8)


# ============================================================================
# PART 8: MAIN INTERFACE
# ============================================================================


def run_algorithm_1(args, env):
    
    print("="*70)
    print("ALGORITHM 1: Discretization Routine")
    print("="*70)
    print(f"Environment: {type(env).__name__}")
    
    print(f"Initializing grid with resolution {args.resolution}^3...")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.resolution)
    reachability = GronwallReachabilityAnalyzer(env)
    value_iter = SafetyValueIterator(env=env, gamma=args.gamma, cell_tree=cell_tree, reachability=reachability,output_dir=f"./results/algorithm1_dynamics_{args.dynamics}_resol_{args.resolution},tol_{args.tolerance:.1e}_tau_{args.tau:.3f}_dt_{args.dt:.3f}_")
    start_time = time.time()
    conv_upper, conv_lower = value_iter.value_iteration(
        max_iterations=args.iterations,
        convergence_tol=args.tolerance,
        plot_freq=args.plot_freq,
        conservative_mode=args.conservative,
        delta_max=args.delta_max
    )
    elapsed = time.time() - start_time
    print(f"\nALGORITHM 1 COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Results saved to: {value_iter.output_dir}/")


def run_algorithm_2(args, env):
    print("="*70)
    print("ALGORITHM 2/3: Adaptive Refinement")
    print("="*70)
    print(f"Environment: {type(env).__name__}")
    # L_f, _ = env.get_lipschitz_constants()
    # if args.gamma * L_f >= 1:
    #     raise ValueError(f"Contraction condition violated: γL_f = {args.gamma * L_f} >= 1")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.initial_resolution)
    reachability = GronwallReachabilityAnalyzer(env)
    adaptive = AdaptiveRefinement(args, env, args.gamma, cell_tree, reachability)
    start_time = time.time()
    adaptive.refine(
        epsilon=args.epsilon,
        max_refinements=args.refinements,
        vi_iterations_per_refinement=args.vi_iterations
    )
    elapsed = time.time() - start_time
    print(f"ALGORITHM 2 COMPLETE")
    print(f"Total time: {elapsed:.2f} seconds")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Safety Value Function Computation - Fixed Implementation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--algorithm', type=int, choices=[1, 2], required=True,
                       help="Algorithm to run: 1 (basic discretization) or 2 (adaptive refinement)")
    
    # ✅ New argument for choosing dynamics
    parser.add_argument('--dynamics', type=str, default='dubins', choices=['dubins', 'evasion'],
                        help="Choose dynamics model: 'dubins' or 'evasion'")
    
    # Environment parameters
    parser.add_argument('--velocity', type=float, default=1.0,
                       help="Constant velocity")
    parser.add_argument('--dt', type=float, default=0.1,
                       help="Checkpoint interval for reachability computation")
    parser.add_argument('--tau', type=float, default=1.0,
                       help="Control duration (action hold time)")
    # parser.add_argument('--obstacle-radius', type=float, default=1.3,
    #                    help="Radius of circular obstacle (Dubins only)")
    parser.add_argument('--gamma', type=float, default=0.1,
                       help="Discount factor (must satisfy γL_f < 1)")
    
    # Algorithm parameters
    parser.add_argument('--resolution', type=int, default=10,
                       help="Grid resolution per dimension (Algorithm 1)")
    parser.add_argument('--iterations', type=int, default=200,
                       help="Maximum value iterations (Algorithm 1)")
    parser.add_argument('--tolerance', type=float, default=1e-13,
                       help="Convergence tolerance")
    parser.add_argument('--plot-freq', type=int, default=100, #dont create intermediate plots
                       help="Plot frequency in iterations (Algorithm 1)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="Error tolerance for refinement (Algorithm 2)")
    parser.add_argument('--initial-resolution', type=int, default=15,
                       help="Initial coarse grid resolution (Algorithm 2)")
    parser.add_argument('--refinements', type=int, default=100,
                       help="Maximum refinement iterations (Algorithm 2)")
    parser.add_argument('--vi-iterations', type=int, default=50,
                       help="VI iterations per refinement (Algorithm 2)")
    parser.add_argument('--conservative', action='store_true', default=False,
                       help="Use conservative stopping condition (Algorithm 3)")
    parser.add_argument('--delta-max', type=float, default=1e-6,
                       help="Maximum allowed decrease δ_max for conservative stopping")
    parser.add_argument('--workers', type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument('--precompute', action='store_true',
                       help="Precompute all successor sets before VI")
    
    args = parser.parse_args()
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    print(f"OPTIMIZED SAFETY VALUE FUNCTION - FIXED IMPLEMENTATION")
    print(f"Algorithm: {args.algorithm}, Workers: {args.workers}")
    print(f"Dynamics: {args.dynamics}")
    print(f"Conservative mode: {args.conservative}")

    # ✅ Environment selection
    if args.dynamics == 'dubins':
        env = DubinsCarEnvironment(
            v_const=args.velocity, dt=args.dt, tau=args.tau, obstacle_radius=1.3
        )
    else:
        env = EvasionEnvironment(
            v_const=args.velocity, dt=args.dt, tau=args.tau,
            obstacle_position=(0.0, 0.0), obstacle_radius=1.0
        )

    # Continue as before
    if args.algorithm == 1:
        run_algorithm_1(args, env)
    else:
        run_algorithm_2(args, env)


if __name__ == "__main__":
    main()
    
# caffeinate -id python -u safety_value_function_4_ris_all_leaves_optimized_new_newstopcondition_dubins_aircraft.py \
# --algorithm 2 \
#     --resolution 50 \
#         --iterations 2002 \
#             --gamma  0.9 --dt 0.2 \
#                 --tau 0.6 --tolerance 1e-13 \
#                     --initial-resolution 2 --vi-iterations 100001\
#                         --eps 0.05 --dynamics dubins  --conservative  \
#                             --delta-max 1e-13 \
#                                  2>&1 | tee ./run_log_dubins.txt
                                 
#  caffeinate -id  python -u safety_value_function_4_ris_all_leaves_optimized_new_newstopcondition_dubins_aircraft.py \
# --algorithm 2 \
#     --resolution 50 \
#         --iterations 2002 \
#             --gamma  0.3 --dt 0.25 \
#                 --tau 0.5 --tolerance 0.000000000000001 \
#                     --initial-resolution 2 --vi-iterations 100001\
#                         --eps 0.05 --dynamics evasion  --conservative  \
#                             --delta-max 1e-13 \
#                                  2>&1 | tee ./run_log_evasion.txt
