"""
Safety Value Functions with Formal Guarantees - Complete Implementation
========================================================================

Implements discrete safety value functions with upper and lower bounds as described in:
"From continuous to discrete state spaces for safety value functions"

Key Features:
- Algorithm 1: Basic discretization with value iteration
- Algorithm 2/3: Adaptive refinement for efficiency
- Formal guarantees via Lipschitz continuity
- Visualization of value functions and cell classifications

Usage:
    python safety_value_function.py --algorithm 1 --resolution 10 --iterations 100
    python safety_value_function.py --algorithm 2 --epsilon 0.1 --refinements 5

Author: Implementation based on theory from the paper
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Patch
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import argparse
import time
import os
from itertools import product
import matplotlib.colors as mcolors


# ============================================================================
# PART 1: ENVIRONMENT DEFINITIONS
# ============================================================================

class Environment(ABC):
    """
    Abstract base class for dynamics environments.
    
    Any environment must define:
    - State space bounds
    - Discrete action space
    - Dynamics function f(x,u)
    - Failure/safety function l(x)
    - Lipschitz constants L_f and L_l
    """
    
    @abstractmethod
    def get_state_bounds(self) -> np.ndarray:
        """Returns state space bounds as (d, 2) array: [[x_min, x_max], [y_min, y_max], ...]"""
        pass
    
    @abstractmethod
    def get_action_space(self) -> List:
        """Returns list of discrete actions."""
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action) -> np.ndarray:
        """Computes next state: x_{t+1} = f(x_t, u_t)"""
        pass
    
    @abstractmethod
    def failure_function(self, state: np.ndarray) -> float:
        """Safety signal l(x): negative = unsafe, positive = safe"""
        pass
    
    @abstractmethod
    def get_lipschitz_constants(self) -> Tuple[float, float]:
        """Returns (L_f, L_l): Lipschitz constants for dynamics and failure function"""
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        """Returns dimension of state space"""
        pass


class DubinsCarEnvironment(Environment):
    """
    Dubins car with constant velocity navigating around a circular obstacle.
    
    State: [x, y, theta] where (x,y) is position and theta is heading angle
    Action: dtheta in {-1, 0, +1} (turn left, straight, turn right)
    Dynamics: x' = x + v*cos(theta)*dt, y' = y + v*sin(theta)*dt, theta' = theta + dtheta*dt
    
    Obstacle: Circular obstacle at specified position with given radius
    Safety: l(x) = distance_to_obstacle - obstacle_radius (negative = collision)
    """
    
    def __init__(
        self,
        v_const: float = 1.0,
        dt: float = 0.1,
        state_bounds: np.ndarray = None,
        obstacle_position: np.ndarray = None,
        obstacle_radius: float = 0.5
    ):
        """
        Initialize Dubins car environment.
        
        Args:
            v_const: Constant linear velocity
            dt: Time step for Euler integration
            state_bounds: State space bounds (3x2), default [-3,3] x [-3,3] x [-π,π]
            obstacle_position: [x, y] center of obstacle, default [0, 0]
            obstacle_radius: Radius of circular obstacle
        """
        self.v_const = v_const
        self.dt = dt
        
        # Default state bounds: x, y in [-3, 3], theta in [-π, π]
        if state_bounds is None:
            self.state_bounds = np.array([
                [-3.0, 3.0],      # x bounds
                [-3.0, 3.0],      # y bounds
                [-np.pi, np.pi]   # theta bounds
            ])
        else:
            self.state_bounds = state_bounds
        
        # Default obstacle at origin
        if obstacle_position is None:
            self.obstacle_position = np.array([0.0, 0.0])
        else:
            self.obstacle_position = obstacle_position
            
        self.obstacle_radius = obstacle_radius
        
        # Lipschitz constants (derived from theory):
        # L_f: From Jacobian of dynamics in infinity norm
        #      ∂f/∂θ has max norm of max(v*dt, 1), so L_f = max(1, v*dt)
        # L_l: Gradient of signed distance to circle has norm 1
        self.L_f = max(1.0, v_const * dt)
        self.L_l = 1.0
        
        # Discrete action space: turn left (-1), straight (0), turn right (+1)
        self.actions = [-1.0, 0.0, 1.0]
    
    def get_state_bounds(self) -> np.ndarray:
        return self.state_bounds
    
    def get_action_space(self) -> List:
        return self.actions
    
    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        """
        Dubins car dynamics with Euler integration.
        
        Args:
            state: [x, y, theta]
            action: dtheta (angular velocity)
            
        Returns:
            Next state [x', y', theta']
        """
        x, y, theta = state
        dtheta = action
        
        # Euler step
        x_next = x + self.v_const * np.cos(theta) * self.dt
        y_next = y + self.v_const * np.sin(theta) * self.dt
        theta_next = theta + dtheta * self.dt
        
        # Wrap theta to [-π, π]
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])
    
    def failure_function(self, state: np.ndarray) -> float:
        """
        Signed distance to obstacle boundary.
        
        Args:
            state: [x, y, theta]
            
        Returns:
            l(x) = ||[x,y] - obstacle_center|| - obstacle_radius
            Negative means inside obstacle (unsafe)
        """
        pos = state[:2]  # Extract [x, y]
        dist_to_obstacle = np.linalg.norm(pos - self.obstacle_position)
        return dist_to_obstacle - self.obstacle_radius
    
    def get_lipschitz_constants(self) -> Tuple[float, float]:
        return self.L_f, self.L_l
    
    def get_state_dim(self) -> int:
        return 3


# ============================================================================
# PART 2: CELL STRUCTURE FOR ADAPTIVE GRIDS
# ============================================================================

class Cell:
    """
    Represents a hyperrectangular cell in the discretized state space.
    
    Each cell stores:
    - Spatial bounds (hyperrectangle)
    - Center point
    - Stage cost bounds: l_lower, l_upper
    - Value function bounds: V_lower, V_upper
    - Tree structure: children (for adaptive refinement)
    """
    
    def __init__(self, bounds: np.ndarray, cell_id: int = 0):
        """
        Initialize a cell.
        
        Args:
            bounds: Array of shape (d, 2) where bounds[j] = [a_j, b_j]
            cell_id: Unique identifier for this cell
        """
        self.bounds = bounds  # (d, 2) array
        self.cell_id = cell_id
        
        # Compute center as midpoint of bounds
        self.center = np.mean(bounds, axis=1)
        
        # Value function bounds (set during value iteration)
        self.V_upper = None  # V̄_γ(s)
        self.V_lower = None  # V_γ(s)
        
        # Stage cost bounds (set during initialization)
        self.l_upper = None  # l̄(s)
        self.l_lower = None  # l(s)
        
        # Tree structure for adaptive refinement
        self.children = []
        self.is_leaf = True
        self.is_refined = False
        
    def get_range(self, dim: int) -> float:
        """Returns the range (width) of the cell along dimension dim."""
        return self.bounds[dim, 1] - self.bounds[dim, 0]
    
    def get_max_range_dim(self) -> int:
        """Returns the dimension index with the largest range."""
        ranges = [self.get_range(j) for j in range(len(self.bounds))]
        return np.argmax(ranges)
    
    def get_max_range(self) -> float:
        """Returns the maximum range across all dimensions."""
        dim = self.get_max_range_dim()
        return self.get_range(dim)
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Checks if a point lies inside this cell."""
        for j in range(len(point)):
            if point[j] < self.bounds[j, 0] or point[j] > self.bounds[j, 1]:
                return False
        return True
    
    def intersects(self, other_bounds: np.ndarray) -> bool:
        """
        Checks if this cell intersects with another hyperrectangle.
        
        Args:
            other_bounds: Bounds of another hyperrectangle (d, 2)
            
        Returns:
            True if there is any overlap
        """
        for j in range(len(self.bounds)):
            # No overlap if this cell's max < other's min OR this cell's min > other's max
            if self.bounds[j, 1] < other_bounds[j, 0] or self.bounds[j, 0] > other_bounds[j, 1]:
                return False
        return True
    
    def split(self, next_id: int) -> Tuple['Cell', 'Cell']:
        """
        Splits this cell into two children along its longest dimension.
        
        Args:
            next_id: Starting ID for new cells
            
        Returns:
            Tuple of two child cells
        """
        # Find dimension with largest extent
        dim = self.get_max_range_dim()
        mid = (self.bounds[dim, 0] + self.bounds[dim, 1]) / 2.0
        
        # Create two children by bisecting along dim
        bounds1 = self.bounds.copy()
        bounds1[dim, 1] = mid  # Left/lower child
        
        bounds2 = self.bounds.copy()
        bounds2[dim, 0] = mid  # Right/upper child
        
        child1 = Cell(bounds1, next_id)
        child2 = Cell(bounds2, next_id + 1)
        
        # Update tree structure
        self.children = [child1, child2]
        self.is_leaf = False
        
        return child1, child2
    
    def __repr__(self):
        return (f"Cell(id={self.cell_id}, center={self.center}, "
                f"V_upper={self.V_upper}, V_lower={self.V_lower})")


class CellTree:
    """
    Tree structure to manage cells with adaptive refinement.
    
    Maintains a collection of leaf cells (active cells in the discretization).
    Supports:
    - Initial uniform grid creation
    - Adaptive cell splitting
    - Querying cells by spatial location
    """
    
    def __init__(self, initial_bounds: np.ndarray, initial_resolution: int = 10):
        """
        Initialize cell tree with uniform grid.
        
        Args:
            initial_bounds: State space bounds (d, 2)
            initial_resolution: Number of cells per dimension initially
        """
        self.root_bounds = initial_bounds
        self.dim = len(initial_bounds)
        self.next_id = 0
        self.leaves = []  # List of all leaf (active) cells
        self._create_initial_grid(initial_resolution)
    
    def _create_initial_grid(self, resolution: int):
        """
        Creates a uniform initial grid.
        
        Args:
            resolution: Number of cells per dimension (total = resolution^dim)
        """
        # For each dimension, create evenly-spaced partition points
        ranges = []
        for j in range(self.dim):
            a, b = self.root_bounds[j]
            ranges.append(np.linspace(a, b, resolution + 1))
        
        # Create cells from Cartesian product of intervals
        indices = [range(resolution) for _ in range(self.dim)]
        for idx in product(*indices):
            bounds = np.zeros((self.dim, 2))
            for j in range(self.dim):
                bounds[j, 0] = ranges[j][idx[j]]
                bounds[j, 1] = ranges[j][idx[j] + 1]
            
            cell = Cell(bounds, self.next_id)
            self.next_id += 1
            self.leaves.append(cell)
    
    def get_leaves(self) -> List[Cell]:
        """Returns all leaf (active) cells."""
        return self.leaves
    
    def refine_cell(self, cell: Cell):
        """
        Refines a cell by splitting it and updating the leaf list.
        
        Args:
            cell: Cell to refine (must be a leaf)
        """
        if not cell.is_leaf:
            return  # Already refined
        
        # Split the cell into two children
        child1, child2 = cell.split(self.next_id)
        self.next_id += 2
        
        # Update leaf list: remove parent, add children
        self.leaves.remove(cell)
        self.leaves.extend([child1, child2])
        
        cell.is_refined = True
    
    def get_cell_containing_point(self, point: np.ndarray) -> Optional[Cell]:
        """
        Finds the leaf cell containing the given point.
        
        Args:
            point: Point in state space
            
        Returns:
            Cell containing the point, or None if outside bounds
        """
        for cell in self.leaves:
            if cell.contains_point(point):
                return cell
        return None
    
    def get_num_leaves(self) -> int:
        """Returns the number of leaf (active) cells."""
        return len(self.leaves)


# ============================================================================
# PART 3: REACHABILITY ANALYSIS
# ============================================================================

class ReachabilityAnalyzer:
    """
    Computes forward reachable sets using grid-based over-approximation.
    
    For a cell s and action u, computes:
    - R(s, u): The reachable set (bounding box of all possible next states)
    - Δ(s, u): Successor cells that intersect R(s, u)
    """
    
    def __init__(self, env: Environment, samples_per_dim: int = 5):
        """
        Args:
            env: Environment defining dynamics
            samples_per_dim: Number of sample points per dimension within each cell
        """
        self.env = env
        self.samples_per_dim = samples_per_dim
    
    def compute_reachable_set(self, cell: Cell, action) -> np.ndarray:
        """
        Computes the bounding box of the reachable set R(s, u).
        
        Method: Samples points within the cell, applies dynamics,
        and computes the axis-aligned bounding box of next states.
        
        Args:
            cell: Starting cell
            action: Control action
            
        Returns:
            Bounds of reachable set as array of shape (d, 2)
        """
        # Sample points uniformly within the cell
        samples = self._sample_cell(cell)
        
        # Apply dynamics to all samples
        next_states = []
        for state in samples:
            next_state = self.env.dynamics(state, action)
            next_states.append(next_state)
        
        next_states = np.array(next_states)
        
        # Compute axis-aligned bounding box
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        reach_bounds[:, 0] = np.min(next_states, axis=0)
        reach_bounds[:, 1] = np.max(next_states, axis=0)
        
        return reach_bounds
    
    def compute_successor_cells(
        self, 
        cell: Cell, 
        action, 
        cell_tree: CellTree
    ) -> List[Cell]:
        """
        Computes the set Δ(s, u) of successor cells.
        
        A cell s' is a successor if s' ∩ R(s, u) ≠ ∅.
        
        Args:
            cell: Starting cell
            action: Control action
            cell_tree: Tree containing all cells
            
        Returns:
            List of successor cells
        """
        # Compute reachable set bounds
        reach_bounds = self.compute_reachable_set(cell, action)
        
        # Find all cells that intersect with reachable set
        successors = []
        for candidate in cell_tree.get_leaves():
            if candidate.intersects(reach_bounds):
                successors.append(candidate)
        
        return successors
    
    def _sample_cell(self, cell: Cell) -> np.ndarray:
        """
        Samples points uniformly within a cell.
        
        Args:
            cell: Cell to sample
            
        Returns:
            Array of shape (num_samples, d) containing sample points
        """
        # For each dimension, create samples_per_dim evenly spaced points
        dim_samples = []
        for j in range(len(cell.bounds)):
            a, b = cell.bounds[j]
            dim_samples.append(np.linspace(a, b, self.samples_per_dim))
        
        # Create Cartesian product
        samples = []
        for point in product(*dim_samples):
            samples.append(np.array(point))
        
        return np.array(samples)


class LipschitzReachabilityAnalyzer(ReachabilityAnalyzer):
    """
    Enhanced reachability using Lipschitz bounds for tighter approximations.
    
    Uses the Lipschitz continuity of dynamics to compute:
    R(s, u) ⊆ {f(x_c, u)} ⊕ L_f * η * Ball_∞
    
    where x_c is the cell center and η is half the max cell width.
    This is faster and often more accurate than sampling.
    """
    
    def compute_reachable_set(self, cell: Cell, action) -> np.ndarray:
        """
        Computes reachable set using center + Lipschitz expansion.
        
        Args:
            cell: Starting cell
            action: Control action
            
        Returns:
            Bounds of reachable set as array of shape (d, 2)
        """
        # Apply dynamics to cell center
        center = cell.center
        center_next = self.env.dynamics(center, action)
        
        # Get Lipschitz constant
        L_f, _ = self.env.get_lipschitz_constants()
        
        # Compute η: maximum deviation from center in infinity norm
        # For a hyperrectangle, η = half the maximum side length
        eta = 0.5 * cell.get_max_range()
        
        # Expand by L_f * η in all directions (infinity ball)
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        expansion = L_f * eta
        
        for j in range(self.env.get_state_dim()):
            reach_bounds[j, 0] = center_next[j] - expansion
            reach_bounds[j, 1] = center_next[j] + expansion
        
        return reach_bounds


# ============================================================================
# PART 4: VALUE ITERATION (ALGORITHM 1)
# ============================================================================

class SafetyValueIterator:
    """
    Implements Algorithm 1: Discretization Routine.
    
    Computes upper and lower bounds on the safety value function:
    - V̄_γ(s): Upper bound on Vγ(x) for all x in cell s
    - V_γ(s): Lower bound on Vγ(x) for all x in cell s
    
    Uses the Bellman operator:
    V̄_γ(s) = min{ l̄(s), max_u max_{s'∈Δ(s,u)} γ V̄_γ(s') }
    V_γ(s) = min{ l(s), max_u min_{s'∈Δ(s,u)} γ V_γ(s') }
    """
    
    def __init__(self,
                 env: Environment,
                 gamma: float,
                 cell_tree: CellTree,
                 reachability: ReachabilityAnalyzer,
                 output_dir: Optional[str] = None):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability

        # Default output dir depends on reachability type
        if output_dir is None:
            rname = type(reachability).__name__
            output_dir = f"./results/{rname}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.L_f, self.L_l = env.get_lipschitz_constants()
        if gamma * self.L_f >= 1:
            raise ValueError(f"Contraction condition violated: γL_f = {gamma * self.L_f} >= 1")

        print(f"Initialized with γ={gamma}, L_f={self.L_f}, L_l={self.L_l}")
        print(f"Contraction factor: γL_f = {gamma * self.L_f}")
    
    def initialize_cells(self):
        """
        Initializes stage cost bounds l(s) and l̄(s) for all cells.
        
        Uses Lipschitz continuity of l:
        l(s) = l(x_c) - L_l * η
        l̄(s) = l(x_c) + L_l * η
        
        where x_c is the cell center and η is the maximum deviation
        from center in infinity norm.
        """
        for cell in self.cell_tree.get_leaves():
            # Evaluate l at center
            l_center = self.env.failure_function(cell.center)
            
            # Compute η = half the maximum cell width (infinity norm)
            # For a hyperrectangle, this is half the longest side
            eta = 0.5 * cell.get_max_range()
            
            # Lipschitz bounds
            cell.l_lower = l_center - self.L_l * eta
            cell.l_upper = l_center + self.L_l * eta
            
            # Initialize value functions to ensure first update is correct
            # Start with -∞ (no safety guarantee) and ∞ (no failure proven)
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
    
    def initialize_new_cells(self, new_cells: List[Cell]):
        """
        Initializes only the newly created cells (used in Algorithm 2).
        
        Args:
            new_cells: List of cells to initialize
        """
        for cell in new_cells:
            l_center = self.env.failure_function(cell.center)
            eta = 0.5 * cell.get_max_range()
            cell.l_lower = l_center - self.L_l * eta
            cell.l_upper = l_center + self.L_l * eta
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
    
    def bellman_update(self, cell: Cell) -> Tuple[float, float]:
        """
        Performs one Bellman update for a cell.
        
        Computes:
        V̄_γ(s) = min{ l̄(s), max_u max_{s'∈Δ(s,u)} γ V̄_γ(s') }
        V_γ(s) = min{ l(s), max_u min_{s'∈Δ(s,u)} γ V_γ(s') }
        
        Args:
            cell: Cell to update
            
        Returns:
            (new_V_upper, new_V_lower)
        """
        max_upper = -np.inf
        max_lower = -np.inf
        
        # Loop over all actions
        for action in self.env.get_action_space():
            # Compute successor cells Δ(s, u)
            successors = self.reachability.compute_successor_cells(
                cell, action, self.cell_tree
            )
            
            if len(successors) == 0:
                continue  # No reachable successors for this action
            
            # Upper bound: take maximum over successors
            upper_vals = [s.V_upper for s in successors if s.V_upper is not None]
            if upper_vals:
                action_upper = self.gamma * max(upper_vals)
                max_upper = max(max_upper, action_upper)
            
            # Lower bound: take minimum over successors
            lower_vals = [s.V_lower for s in successors if s.V_lower is not None]
            if lower_vals:
                action_lower = self.gamma * min(lower_vals)
                max_lower = max(max_lower, action_lower)
        
        # Apply min with stage cost (terminal condition)
        new_V_upper = min(cell.l_upper, max_upper) if max_upper > -np.inf else cell.l_upper
        new_V_lower = min(cell.l_lower, max_lower) if max_lower > -np.inf else cell.l_lower
        
        return new_V_upper, new_V_lower
    
    def value_iteration(
        self, 
        max_iterations: int = 1000,
        convergence_tol: float = 1e-3,
        plot_freq: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs value iteration until convergence.
        
        Algorithm:
        1. Initialize all cells with stage cost bounds
        2. Repeat until convergence:
           - For each cell, apply Bellman update
           - Check ||V^k - V^{k-1}||_∞ < tolerance
        3. Save plots periodically
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_tol: Convergence tolerance for ||V^k - V^{k-1}||_∞
            plot_freq: Frequency (in iterations) to save plots
            
        Returns:
            (convergence_history_upper, convergence_history_lower)
        """
        # Initialize all cells
        self.initialize_cells()
        
        conv_history_upper = []
        conv_history_lower = []
        
        print(f"\nStarting value iteration (max {max_iterations} iterations)...")
        print(f"Convergence tolerance: {convergence_tol}")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        
        for iteration in range(max_iterations):
            # Store previous values for convergence check
            prev_upper = {cell.cell_id: cell.V_upper for cell in self.cell_tree.get_leaves()}
            prev_lower = {cell.cell_id: cell.V_lower for cell in self.cell_tree.get_leaves()}
            
            # Apply Bellman update to all cells (synchronous update)
            updates = {}
            for cell in self.cell_tree.get_leaves():
                new_upper, new_lower = self.bellman_update(cell)
                updates[cell.cell_id] = (new_upper, new_lower)
            
            # Apply updates simultaneously
            for cell in self.cell_tree.get_leaves():
                cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            # Compute convergence metrics (infinity norm)
            diff_upper = max(
                abs(cell.V_upper - prev_upper[cell.cell_id])
                for cell in self.cell_tree.get_leaves()
            )
            diff_lower = max(
                abs(cell.V_lower - prev_lower[cell.cell_id])
                for cell in self.cell_tree.get_leaves()
            )
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            # Print progress
            print(f"Iteration {iteration + 1}: "
                  f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.6f}, "
                  f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.6f}")
            
            # Save plot periodically
            if (iteration + 1) % plot_freq == 0:
                self._save_plot(iteration + 1)
            
            # Check convergence
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"\n✓ Converged at iteration {iteration + 1}!")
                self._save_plot(iteration + 1, final=True)
                break
        else:
            print(f"\nReached maximum iterations ({max_iterations})")
            self._save_plot(max_iterations, final=True)
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
    
    def _continue_value_iteration(
        self,
        max_iterations: int,
        convergence_tol: float = 1e-3,
        plot_freq: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Continues value iteration WITHOUT reinitializing cells.
        Used in Algorithm 2 after adaptive refinement.
        
        This differs from value_iteration() by NOT calling initialize_cells(),
        so existing V values are preserved.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_tol: Convergence tolerance
            plot_freq: How often to save plots
            
        Returns:
            (convergence_history_upper, convergence_history_lower)
        """
        conv_history_upper = []
        conv_history_lower = []
        
        print(f"Continuing value iteration (max {max_iterations} iterations)...")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        
        for iteration in range(max_iterations):
            # Store previous values
            prev_upper = {cell.cell_id: cell.V_upper for cell in self.cell_tree.get_leaves()}
            prev_lower = {cell.cell_id: cell.V_lower for cell in self.cell_tree.get_leaves()}
            
            # Apply Bellman update
            updates = {}
            for cell in self.cell_tree.get_leaves():
                new_upper, new_lower = self.bellman_update(cell)
                updates[cell.cell_id] = (new_upper, new_lower)
            
            # Apply updates
            for cell in self.cell_tree.get_leaves():
                cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            # Compute convergence
            diff_upper = max(
                abs(cell.V_upper - prev_upper[cell.cell_id])
                for cell in self.cell_tree.get_leaves()
            )
            diff_lower = max(
                abs(cell.V_lower - prev_lower[cell.cell_id])
                for cell in self.cell_tree.get_leaves()
            )
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"Iteration {iteration + 1}: "
                  f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.6f}, "
                  f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.6f}")
            
            if (iteration + 1) % plot_freq == 0:
                self._save_plot(iteration + 1)
            
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"Converged at iteration {iteration + 1}!")
                break
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
    
    def _save_plot(self, iteration: int, final: bool = False):
        """Saves visualization of current value function."""
        suffix = "_final" if final else ""
        filename = os.path.join(self.output_dir, f"iteration_{iteration:04d}{suffix}.png")
        plot_value_function(self.env, self.cell_tree, filename, iteration)
        if final:
            print(f"Final plot saved to {filename}")


# ============================================================================
# PART 5: ADAPTIVE REFINEMENT (ALGORITHM 2/3)
# ============================================================================

class AdaptiveRefinement:
    """
    Implements Algorithm 2/3: Adaptive refinement of boundary cells.
    
    Strategy:
    1. Start with coarse grid and run value iteration
    2. Identify boundary cells (where V̄_γ > 0 and V_γ < 0)
    3. Refine boundary cells by splitting
    4. Reinitialize only new cells
    5. Continue value iteration (without resetting existing cells)
    6. Repeat until error tolerance met or max refinements reached
    
    This focuses computational effort on the safety boundary while
    keeping safe and unsafe regions coarse.
    """
    
    def __init__(
        self,
        env: Environment,
        gamma: float,
        cell_tree: CellTree,
        reachability: ReachabilityAnalyzer,
        output_dir: str = "./results_adaptive"
    ):
        """
        Initialize adaptive refinement.
        
        Args:
            env: Environment
            gamma: Discount factor
            cell_tree: Initial cell tree (typically coarse)
            reachability: Reachability analyzer
            output_dir: Output directory for results
        """
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.output_dir = output_dir
        
        # Create value iterator
        self.value_iterator = SafetyValueIterator(
            env, gamma, cell_tree, reachability, output_dir
        )
        
        # Get Lipschitz constant for l
        self.L_l = env.get_lipschitz_constants()[1]
    
    def refine(
        self,
        epsilon: float,
        max_refinements: int = 100,
        vi_iterations_per_refinement: int = 100
    ):
        """
        Main adaptive refinement loop (Algorithm 2/3).
        
        Args:
            epsilon: Error tolerance (discretization error bound)
            max_refinements: Maximum number of refinement iterations
            vi_iterations_per_refinement: VI iterations after each refinement
        """
        # Compute minimum cell size from error tolerance
        # From theory: discretization error ≤ 2*L_l*η, so η_min = ε/(2*L_l)
        eta_min = epsilon / (2 * self.L_l)
        print(f"\nAdaptive refinement with ε={epsilon}, η_min={eta_min:.4f}")
        print(f"Cells smaller than η_min will not be refined further.")
        
        # Initial value iteration on coarse grid
        print("\n" + "="*70)
        print("INITIAL VALUE ITERATION")
        print("="*70)
        self.value_iterator.value_iteration(
            max_iterations=vi_iterations_per_refinement,
            plot_freq=20
        )
        
        # Adaptive refinement loop
        refinement_iter = 0
        while refinement_iter < max_refinements:
            # Identify cells on the boundary
            boundary_cells = self._identify_boundary_cells()
            
            print(f"\n" + "="*70)
            print(f"REFINEMENT ITERATION {refinement_iter + 1}")
            print("="*70)
            print(f"Boundary cells: {len(boundary_cells)}")
            
            if len(boundary_cells) == 0:
                print("No boundary cells remaining - all cells classified!")
                break
            
            # Refine boundary cells that are still too large
            refined_any = False
            new_cells = []
            
            for cell in boundary_cells:
                max_range = cell.get_max_range()
                if max_range > eta_min:
                    # Split this cell
                    self.cell_tree.refine_cell(cell)
                    # Track the newly created children
                    new_cells.extend(cell.children)
                    refined_any = True
            
            if not refined_any:
                print("All boundary cells below minimum resolution η_min!")
                break
            
            print(f"Refined {len(new_cells)//2} cells into {len(new_cells)} new cells")
            print(f"Total cells: {self.cell_tree.get_num_leaves()}")
            
            # Initialize only the new cells (CRITICAL: don't reset existing cells)
            if new_cells:
                print(f"Initializing {len(new_cells)} new cells...")
                self.value_iterator.initialize_new_cells(new_cells)
            
            # Continue value iteration WITHOUT reinitializing existing cells
            print(f"\nContinuing value iteration...")
            self.value_iterator._continue_value_iteration(
                max_iterations=vi_iterations_per_refinement,
                plot_freq=20
            )
            
            refinement_iter += 1
        
        # Final statistics
        print(f"\n" + "="*70)
        print("REFINEMENT COMPLETE")
        print("="*70)
        print(f"Total refinement iterations: {refinement_iter}")
        print(f"Final number of cells: {self.cell_tree.get_num_leaves()}")
        self._print_statistics()
    
    def _identify_boundary_cells(self) -> List[Cell]:
        """
        Identifies boundary cells where V̄_γ(s) > 0 and V_γ(s) < 0.
        
        These cells intersect both safe and unsafe regions and need refinement.
        
        Returns:
            List of boundary cells
        """
        boundary = []
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                # Boundary: upper bound positive, lower bound negative
                if cell.V_upper > 0 and cell.V_lower < 0:
                    boundary.append(cell)
        return boundary
    
    def _print_statistics(self):
        """Prints final statistics about safe/unsafe/boundary regions."""
        safe_cells = 0
        unsafe_cells = 0
        boundary_cells = 0
        
        for cell in self.cell_tree.get_leaves():
            if cell.V_lower is not None and cell.V_upper is not None:
                if cell.V_lower > 0:
                    safe_cells += 1  # Definitely safe
                elif cell.V_upper < 0:
                    unsafe_cells += 1  # Definitely unsafe
                else:
                    boundary_cells += 1  # Boundary (ambiguous)
        
        total = self.cell_tree.get_num_leaves()
        print(f"\nCell Classification:")
        print(f"  Safe cells:     {safe_cells:5d} ({100*safe_cells/total:5.1f}%)")
        print(f"  Unsafe cells:   {unsafe_cells:5d} ({100*unsafe_cells/total:5.1f}%)")
        print(f"  Boundary cells: {boundary_cells:5d} ({100*boundary_cells/total:5.1f}%)")


# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

def plot_value_function(
    env: Environment,
    cell_tree: CellTree,
    filename: str,
    iteration: int,
    theta_slices: list = None
):
    """
    Plots 2D slices of the 3D value function for Dubins car.
    
    Creates a 3-row plot:
    - Row 1: Upper bound V̄_γ at different θ values
    - Row 2: Lower bound V_γ at different θ values
    - Row 3: Cell classification (safe/unsafe/boundary)
    
    Args:
        env: Environment
        cell_tree: Cell tree with computed values
        filename: Output filename
        iteration: Current iteration number
        theta_slices: List of θ values to plot (default: [0, π/4, π/2])
    """
    if theta_slices is None:
        theta_slices = [0, np.pi/4, np.pi/2]
    
    n_slices = len(theta_slices)
    fig, axes = plt.subplots(3, n_slices, figsize=(5*n_slices, 14))
    
    if n_slices == 1:
        axes = axes.reshape(3, 1)
    
    for idx, theta in enumerate(theta_slices):
        # Upper bound
        ax_upper = axes[0, idx]
        _plot_slice(env, cell_tree, theta, ax_upper, "V̄_γ", upper=True)
        ax_upper.set_title(f"Upper Bound V̄_γ (θ={theta:.2f} rad, {np.degrees(theta):.0f}°)")
        
        # Lower bound
        ax_lower = axes[1, idx]
        _plot_slice(env, cell_tree, theta, ax_lower, "V_γ", upper=False)
        ax_lower.set_title(f"Lower Bound V_γ (θ={theta:.2f} rad, {np.degrees(theta):.0f}°)")
        
        # Classification
        ax_cls = axes[2, idx]
        _plot_classification_slice(env, cell_tree, theta, ax_cls)
        ax_cls.set_title(f"Cell Classification (θ={theta:.2f} rad, {np.degrees(theta):.0f}°)")
    
    fig.suptitle(f"Safety Value Function - Iteration {iteration}", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_slice(
    env: Environment,
    cell_tree: CellTree,
    theta: float,
    ax,
    label: str,
    upper: bool
):
    """
    Plots a 2D slice (x-y plane) of the value function at fixed θ.
    
    Shows:
    - Smooth color map of value function
    - Black contour at zero level set (safety boundary)
    - Black cell boundaries
    - Red obstacle circle
    """
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Create dense grid for smooth visualization
    resolution = 100
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Evaluate value function at each grid point
    V = np.full_like(X, np.nan, dtype=float)
    for i in range(resolution):
        for j in range(resolution):
            state = np.array([X[i, j], Y[i, j], theta])
            cell = cell_tree.get_cell_containing_point(state)
            
            if cell is not None:
                if upper and cell.V_upper is not None:
                    V[i, j] = cell.V_upper
                elif not upper and cell.V_lower is not None:
                    V[i, j] = cell.V_lower
    
    # Filled contour for smooth background (green=safe, red=unsafe)
    im = ax.contourf(X, Y, V, levels=20, cmap='RdYlGn', alpha=0.75)
    
    # Draw zero level set (black thick line = safety boundary)
    ax.contour(X, Y, V, levels=[0.0], colors='black', linewidths=2.5)
    
    # Draw cell boundaries
    for cell in cell_tree.get_leaves():
        a_x, b_x = cell.bounds[0]
        a_y, b_y = cell.bounds[1]
        rect = Rectangle(
            (a_x, a_y),
            b_x - a_x,
            b_y - a_y,
            fill=False,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            color='red',
            alpha=0.5,
            zorder=10
        )
        ax.add_patch(obstacle)
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.colorbar(im, ax=ax).set_label(label)


def _plot_classification_slice(
    env: Environment,
    cell_tree: CellTree,
    theta: float,
    ax
):
    """
    Plots cell classification: safe (green), unsafe (red), boundary (gray).
    
    Classification rules (Theorem 3):
    - Safe: V̄_γ > 0 AND V_γ > 0
    - Unsafe: V̄_γ < 0 AND V_γ < 0
    - Boundary: Mixed signs (needs refinement)
    """
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Colors for classification
    C_UNSAFE = "#d62728"  # Red
    C_SAFE = "#2ca02c"    # Green
    C_BOUND = "#7f7f7f"   # Gray
    
    # Draw each cell with appropriate color
    for cell in cell_tree.get_leaves():
        # Only show cells whose θ-interval contains this slice
        if not (cell.bounds[2, 0] <= theta <= cell.bounds[2, 1]):
            continue
        
        if cell.V_upper is None or cell.V_lower is None:
            continue
        
        # Classify cell by sign agreement
        if (cell.V_lower > 0) and (cell.V_upper > 0):
            face_color = C_SAFE  # Definitely safe
        elif (cell.V_lower < 0) and (cell.V_upper < 0):
            face_color = C_UNSAFE  # Definitely unsafe
        else:
            face_color = C_BOUND  # Boundary (ambiguous)
        
        rect = Rectangle(
            (cell.bounds[0, 0], cell.bounds[1, 0]),
            cell.get_range(0),
            cell.get_range(1),
            facecolor=face_color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            color='red',
            alpha=0.5,
            zorder=10
        )
        ax.add_patch(obstacle)
    
    # Formatting
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Legend
    legend_elems = [
        Patch(facecolor=C_SAFE, edgecolor='black', label='Safe (V_>0 & V̄>0)'),
        Patch(facecolor=C_UNSAFE, edgecolor='black', label='Unsafe (V_<0 & V̄<0)'),
        Patch(facecolor=C_BOUND, edgecolor='black', label='Boundary (mixed)')
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8, framealpha=0.9)


def plot_convergence(
    conv_upper: np.ndarray,
    conv_lower: np.ndarray,
    filename: str
):
    """
    Plots convergence history showing geometric contraction.
    
    Args:
        conv_upper: Upper bound convergence history ||V̄^k - V̄^{k-1}||_∞
        conv_lower: Lower bound convergence history ||V_^k - V_^{k-1}||_∞
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, len(conv_upper) + 1)
    
    ax.semilogy(iterations, conv_upper, 'b-', label='||V̄^k - V̄^{k-1}||_∞',
                linewidth=2, marker='o', markersize=4, markevery=max(1, len(iterations)//20))
    ax.semilogy(iterations, conv_lower, 'r-', label='||V_^k - V_^{k-1}||_∞',
                linewidth=2, marker='s', markersize=4, markevery=max(1, len(iterations)//20))
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Infinity Norm Difference', fontsize=12)
    ax.set_title('Value Iteration Convergence (Geometric Contraction)', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {filename}")


def plot_reachability_single_cell(
    env,
    cell_tree,
    reachability,
    cell_idx=None,
    n_samples=10,
    figsize=(7, 7),
    save_dir="./results",
    filename_prefix="reachability"
):
    """
    Visualizes reachable cells from a single source cell for each action.
    
    Shows:
    - Grid of all cells
    - Source cell highlighted
    - Arrows to reachable cells (colored by action)
    - Sample trajectories from points within source cell
    
    Useful for debugging reachability computation.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    n_cells = len(leaves)
    if cell_idx is None:
        cell_idx = n_cells // 2
    
    src_cell = leaves[cell_idx]
    src_center = src_cell.center[:2]
    theta_center = src_cell.center[2]
    
    # Setup figure
    bounds = env.get_state_bounds()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.set_title(
        f"Reachable cells from cell #{cell_idx} "
        f"(θ={theta_center:.2f} rad, {np.degrees(theta_center):.1f}°)"
    )
    
    # Colors for actions
    action_colors = {-1.0: 'blue', 0.0: 'green', 1.0: 'orange'}
    offset_angles = {-1.0: -8*np.pi/180, 0.0: 0.0, 1.0: 8*np.pi/180}
    
    # Draw grid
    for cell in leaves:
        rect = Rectangle(
            (cell.bounds[0, 0], cell.bounds[1, 0]),
            cell.get_range(0),
            cell.get_range(1),
            fill=False,
            edgecolor='gray',
            linewidth=0.4,
            alpha=0.3
        )
        ax.add_patch(rect)
    
    # Highlight source cell
    rect_src = Rectangle(
        (src_cell.bounds[0, 0], src_cell.bounds[1, 0]),
        src_cell.get_range(0),
        src_cell.get_range(1),
        fill=False,
        edgecolor='black',
        linewidth=2.0
    )
    ax.add_patch(rect_src)
    ax.plot(*src_center, 'ko', markersize=6, label='Source cell center')
    
    # Grid-level reachability arrows
    for action in env.get_action_space():
        succ_cells = reachability.compute_successor_cells(src_cell, action, cell_tree)
        color = action_colors[action]
        offset_angle = offset_angles[action]
        
        for dst_cell in succ_cells:
            dst_center = dst_cell.center[:2]
            dx = dst_center[0] - src_center[0]
            dy = dst_center[1] - src_center[1]
            
            # Small angular offset to avoid overlapping arrows
            if offset_angle != 0.0:
                rot = np.array([
                    [np.cos(offset_angle), -np.sin(offset_angle)],
                    [np.sin(offset_angle), np.cos(offset_angle)]
                ])
                dx, dy = rot @ np.array([dx, dy])
            
            ax.arrow(src_center[0], src_center[1], dx, dy,
                    color=color, alpha=0.6, linewidth=2.0,
                    head_width=0.10, length_includes_head=True)
        
        ax.plot([], [], color=color, linewidth=2.0, label=f"Action {action:+.0f}")
    
    # Sample-based trajectories
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.uniform(src_cell.bounds[0, 0], src_cell.bounds[0, 1], n_samples),
        rng.uniform(src_cell.bounds[1, 0], src_cell.bounds[1, 1], n_samples),
        rng.uniform(src_cell.bounds[2, 0], src_cell.bounds[2, 1], n_samples)
    ])
    
    for s in samples:
        x0, y0 = s[:2]
        for action in env.get_action_space():
            color = action_colors[action]
            next_s = env.dynamics(s, action)
            dx = next_s[0] - x0
            dy = next_s[1] - y0
            ax.plot([x0, x0 + dx], [y0, y0 + dy],
                   color=color, alpha=0.9, linewidth=1.8, zorder=5)
        ax.plot(x0, y0, 'ko', markersize=3, alpha=0.7)
    
    # Obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            color='red',
            alpha=0.5,
            zorder=4
        )
        ax.add_patch(obstacle)
    
    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    
    filename = os.path.join(save_dir, f"{filename_prefix}_cell{cell_idx}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved reachability plot to {filename}")


def plot_failure_function_bounds(
    env,
    cell_tree,
    filename_prefix="ell_bounds",
    save_dir="./results"
):
    """
    Visualizes l_lower and l_upper for all cells as two side-by-side heatmaps.
    
    Useful for verifying Lipschitz bounds are computed correctly.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Collect all l values for normalization
    vals_lower = [c.l_lower for c in leaves if c.l_lower is not None]
    vals_upper = [c.l_upper for c in leaves if c.l_upper is not None]
    vmin_lower, vmax_lower = np.min(vals_lower), np.max(vals_lower)
    vmin_upper, vmax_upper = np.min(vals_upper), np.max(vals_upper)
    
    cmap = plt.cm.RdYlGn
    norms = {
        "l_lower": mcolors.Normalize(vmin=vmin_lower, vmax=vmax_lower),
        "l_upper": mcolors.Normalize(vmin=vmin_upper, vmax=vmax_upper),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    titles = [r"$\ell_{\mathrm{lower}}(s)$", r"$\ell_{\mathrm{upper}}(s)$"]
    
    for ax, attr, title in zip(axes, ["l_lower", "l_upper"], titles):
        norm = norms[attr]
        for cell in leaves:
            val = getattr(cell, attr)
            if val is None:
                continue
            
            color = cmap(norm(val))
            rect = Rectangle(
                (cell.bounds[0, 0], cell.bounds[1, 0]),
                cell.get_range(0),
                cell.get_range(1),
                facecolor=color,
                edgecolor="black",
                linewidth=0.4
            )
            ax.add_patch(rect)
            
            # Draw numerical value at cell center
            cx, cy = cell.center[:2]
            color_text = "white" if val < 0 else "black"
            ax.text(cx, cy, f"{val:.2f}", ha="center", va="center",
                   fontsize=6, color=color_text)
        
        # Obstacle
        if isinstance(env, DubinsCarEnvironment):
            obstacle = Circle(
                env.obstacle_position,
                env.obstacle_radius,
                color='red',
                alpha=0.5
            )
            ax.add_patch(obstacle)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(r"$\ell$ value", rotation=270, labelpad=15)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ℓ-bounds plot to {filename}")


# ============================================================================
# PART 7: MAIN INTERFACE
# ============================================================================

def run_algorithm_1(args):
    """Runs Algorithm 1: Basic discretization with value iteration."""
    print("="*70)
    print("ALGORITHM 1: Discretization Routine")
    print("="*70)
    
    # Create environment
    env = DubinsCarEnvironment(
        v_const=args.velocity,
        dt=args.dt,
        obstacle_radius=args.obstacle_radius
    )
    
    print(f"\nEnvironment: Dubins Car")
    print(f"  Velocity: {args.velocity}")
    print(f"  Time step: {args.dt}")
    print(f"  Obstacle radius: {args.obstacle_radius}")
    print(f"  Discount factor γ: {args.gamma}")
    
    # Create cell tree
    print(f"\nInitializing grid with resolution {args.resolution}^3...")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.resolution)
    print(f"  Total cells: {cell_tree.get_num_leaves()}")
    
    # Create reachability analyzer
    if args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using Lipschitz-based reachability (faster)")
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using sampling-based reachability ({args.samples} samples/dim)")
    
    # Create value iterator (this sets output_dir based on reachability type)
    value_iter = SafetyValueIterator(
        env=env,
        gamma=args.gamma,
        cell_tree=cell_tree,
        reachability=reachability,
        output_dir=None  # None triggers automatic naming
    )
    
    # Optional: Plot reachability from a sample cell
    # FIXED: Now uses value_iter.output_dir instead of hardcoded path
    if args.plot_reachability:
        print("\nGenerating reachability visualization...")
        plot_reachability_single_cell(
            env, cell_tree, reachability, 
            cell_idx=45, 
            save_dir=value_iter.output_dir  # ← CHANGE HERE
        )
    
    # Optional: Plot failure function bounds
    # FIXED: Now uses value_iter.output_dir instead of hardcoded path
    if args.plot_failure:
        print("Generating failure function bounds visualization...")
        value_iter.initialize_cells()
        plot_failure_function_bounds(
            env, value_iter.cell_tree, 
            save_dir=value_iter.output_dir  # ← CHANGE HERE
        )
    
    # Run value iteration
    start_time = time.time()
    conv_upper, conv_lower = value_iter.value_iteration(
        max_iterations=args.iterations,
        convergence_tol=args.tolerance,
        plot_freq=args.plot_freq
    )
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n" + "="*70)
    print("ALGORITHM 1 COMPLETE")
    print("="*70)
    print(f"Total computation time: {elapsed:.2f} seconds")
    print(f"Time per iteration: {elapsed/len(conv_upper):.3f} seconds")
    print(f"Total iterations: {len(conv_upper)}")
    
    # Plot convergence
    plot_convergence(conv_upper, conv_lower, f"{value_iter.output_dir}/convergence.png")
    print(f"\nAll results saved to: {value_iter.output_dir}/")  # ← CHANGE HERE TOO

def run_algorithm_2(args):
    """Runs Algorithm 2/3: Adaptive refinement."""
    print("="*70)
    print("ALGORITHM 2/3: Adaptive Refinement")
    print("="*70)
    
    # Create environment
    env = DubinsCarEnvironment(
        v_const=args.velocity,
        dt=args.dt,
        obstacle_radius=args.obstacle_radius
    )
    
    print(f"\nEnvironment: Dubins Car")
    print(f"  Velocity: {args.velocity}")
    print(f"  Time step: {args.dt}")
    print(f"  Obstacle radius: {args.obstacle_radius}")
    print(f"  Discount factor γ: {args.gamma}")
    print(f"  Error tolerance ε: {args.epsilon}")
    
    # Create initial coarse grid
    print(f"\nInitializing coarse grid with resolution {args.initial_resolution}^3...")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.initial_resolution)
    print(f"  Initial cells: {cell_tree.get_num_leaves()}")
    
    # Create reachability analyzer
    if args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using Lipschitz-based reachability")
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using sampling-based reachability")
    
    # Create adaptive refinement
    adaptive = AdaptiveRefinement(
        env=env,
        gamma=args.gamma,
        cell_tree=cell_tree,
        reachability=reachability,
        output_dir=args.output_dir
    )
    
    # Run adaptive refinement
    start_time = time.time()
    adaptive.refine(
        epsilon=args.epsilon,
        max_refinements=args.refinements,
        vi_iterations_per_refinement=args.vi_iterations
    )
    elapsed = time.time() - start_time
    
    # Print summary
    print(f"\n" + "="*70)
    print("ALGORITHM 2 COMPLETE")
    print("="*70)
    print(f"Total computation time: {elapsed:.2f} seconds")
    print(f"\nAll results saved to: {args.output_dir}/")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Safety Value Function Computation with Formal Guarantees",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm selection
    parser.add_argument(
        '--algorithm', type=int, choices=[1, 2], required=True,
        help="Algorithm to run: 1 (basic discretization) or 2 (adaptive refinement)"
    )
    
    # Environment parameters
    parser.add_argument('--velocity', type=float, default=1.0,
                       help="Constant velocity for Dubins car")
    parser.add_argument('--dt', type=float, default=0.1,
                       help="Time step for dynamics integration")
    parser.add_argument('--obstacle-radius', type=float, default=1.3,
                       help="Radius of circular obstacle")
    parser.add_argument('--gamma', type=float, default=0.9,
                       help="Discount factor (must satisfy γL_f < 1)")
    
    # Algorithm 1 parameters
    parser.add_argument('--resolution', type=int, default=10,
                       help="Grid resolution per dimension (Algorithm 1)")
    parser.add_argument('--iterations', type=int, default=200,
                       help="Maximum value iterations (Algorithm 1)")
    parser.add_argument('--tolerance', type=float, default=1e-3,
                       help="Convergence tolerance (Algorithm 1)")
    parser.add_argument('--plot-freq', type=int, default=10,
                       help="Plot frequency in iterations (Algorithm 1)")
    
    # Algorithm 2 parameters
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="Error tolerance for refinement (Algorithm 2)")
    parser.add_argument('--initial-resolution', type=int, default=8,
                       help="Initial coarse grid resolution (Algorithm 2)")
    parser.add_argument('--refinements', type=int, default=5,
                       help="Maximum refinement iterations (Algorithm 2)")
    parser.add_argument('--vi-iterations', type=int, default=100,
                       help="VI iterations per refinement (Algorithm 2)")
    
    # Reachability parameters
    parser.add_argument('--samples', type=int, default=10,
                       help="Samples per dimension for reachability")
    parser.add_argument('--lipschitz-reach', action='store_true',
                       help="Use Lipschitz-based reachability (faster)")
    
    # Visualization options
    parser.add_argument('--plot-reachability', action='store_true',
                       help="Generate reachability visualization (Algorithm 1)")
    parser.add_argument('--plot-failure', action='store_true',
                       help="Generate failure function bounds plot (Algorithm 1)")
    
    # Output
    # parser.add_argument('--output-dir', type=str, default='./results',
                    #    help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run selected algorithm
    if args.algorithm == 1:
        run_algorithm_1(args)
    else:
        run_algorithm_2(args)


if __name__ == "__main__":
    main()


# ============================================================================
# NOTES AND REFERENCE
# ============================================================================
"""
Coordinate system for Dubins car:
              ↑ θ = π/2 (90°)
  θ = π (180°) ←----o----→  θ = 0 (0°)
              ↓ θ = -π/2 (-90°)

Example usage:
    # Algorithm 1 with default parameters
    python safety_value_function.py --algorithm 1 --resolution 10 --iterations 100
    
    # Algorithm 1 with Lipschitz reachability (faster)
    python safety_value_function.py --algorithm 1 --resolution 8 --iterations 100 --lipschitz-reach
    
    # Algorithm 2 with adaptive refinement
    python safety_value_function.py --algorithm 2 --epsilon 0.1 --refinements 5
    
    # Algorithm 1 with diagnostic plots
    python safety_value_function.py --algorithm 1 --resolution 8 --iterations 50 --plot-reachability --plot-failure

Theory summary:
- Contraction: γL_f < 1 ensures geometric convergence (Theorem 1)
- Error bounds: |V(x) - V(cell_center)| ≤ L_l * η (Corollary 2)
- Soundness: V_γ(s) ≤ Vγ(x) ≤ V̄_γ(s) for all x in s (Theorem 3)
- Adaptive refinement minimizes cells at tolerance ε (Algorithm 2/3)
"""


#python safety_value_function_2.py --algorithm 1 --resolution 10 --iterations 100 --lipschitz-reach --plot-failure  --plot-reachability
#python safety_value_function_2.py --algorithm 1 --resolution 10 --iterations 100 --plot-failure  --plot-reachability