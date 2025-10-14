"""
Safety Value Functions with Formal Guarantees - FIXED Implementation
=====================================================================

Key Fixes:
1. Algorithm 1 convergence checks boundary cells only (practical interpretation)
2. Algorithm 2 uses LOCAL value iteration on refined cells + affected neighbors
3. Proper plot generation after each refinement phase
4. Cell classification uses tree traversal to find correct leaf cells

Usage:
    python safety_value_function_fixed.py --algorithm 1 --resolution 10 --iterations 100
    python safety_value_function_fixed.py --algorithm 2 --epsilon 0.1 --refinements 5
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


# ============================================================================
# PART 1: ENVIRONMENT DEFINITIONS
# ============================================================================

class Environment(ABC):
    """Abstract base class for dynamics environments."""
    
    @abstractmethod
    def get_state_bounds(self) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_action_space(self) -> List:
        pass
    
    @abstractmethod
    def dynamics(self, state: np.ndarray, action) -> np.ndarray:
        pass
    
    @abstractmethod
    def failure_function(self, state: np.ndarray) -> float:
        pass
    
    @abstractmethod
    def get_lipschitz_constants(self) -> Tuple[float, float]:
        pass
    
    @abstractmethod
    def get_state_dim(self) -> int:
        pass


class DubinsCarEnvironment(Environment):
    """Dubins car with constant velocity navigating around a circular obstacle."""
    
    def __init__(self, v_const: float = 1.0, dt: float = 0.1,
                 state_bounds: np.ndarray = None, obstacle_position: np.ndarray = None,
                 obstacle_radius: float = 0.5):
        self.v_const = v_const
        self.dt = dt
        
        if state_bounds is None:
            self.state_bounds = np.array([[-3.0, 3.0], [-3.0, 3.0], [-np.pi, np.pi]])
        else:
            self.state_bounds = state_bounds
        
        if obstacle_position is None:
            self.obstacle_position = np.array([0.0, 0.0])
        else:
            self.obstacle_position = obstacle_position
            
        self.obstacle_radius = obstacle_radius
        self.L_f = 1.0 + v_const * dt
        self.L_l = 1.0
        self.actions = [-1.0, 0.0, 1.0]
    
    def get_state_bounds(self) -> np.ndarray:
        return self.state_bounds
    
    def get_action_space(self) -> List:
        return self.actions
    
    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        x, y, theta = state
        dtheta = action
        
        x_next = x + self.v_const * np.cos(theta) * self.dt
        y_next = y + self.v_const * np.sin(theta) * self.dt
        theta_next = theta + dtheta * self.dt
        theta_next = np.arctan2(np.sin(theta_next), np.cos(theta_next))
        
        return np.array([x_next, y_next, theta_next])
    
    def failure_function(self, state: np.ndarray) -> float:
        pos = state[:2]
        dist_to_obstacle = np.linalg.norm(pos - self.obstacle_position)
        return dist_to_obstacle - self.obstacle_radius
    
    def get_lipschitz_constants(self) -> Tuple[float, float]:
        return self.L_f, self.L_l
    
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
        
        self.V_upper = None
        self.V_lower = None
        self.l_upper = None
        self.l_lower = None
        
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
    """Tree structure to manage cells with adaptive refinement."""
    
    def __init__(self, initial_bounds: np.ndarray, initial_resolution: int = 10):
        self.root_bounds = initial_bounds
        self.dim = len(initial_bounds)
        self.next_id = 0
        self.leaves = []
        self.all_cells = []  # Keep track of all cells (including non-leaves)
        self._create_initial_grid(initial_resolution)
    
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
    
    def get_cell_containing_point(self, point: np.ndarray) -> Optional[Cell]:
        """Find the LEAF cell containing the point by traversing the tree."""
        for cell in self.leaves:
            if cell.contains_point(point):
                return cell
        return None
    
    def get_num_leaves(self) -> int:
        return len(self.leaves)

class GronwallReachabilityAnalyzer:
    """
    Reachability using Grönwall's inequality: expansion = r * e^(Lt)
    
    FIXED: Uses robust angle interval handling without wraparound logic.
    """
    
    def __init__(self, env, use_infinity_norm: bool = True):
        """
        Args:
            env: Environment with dynamics and Lipschitz constants
            use_infinity_norm: If True, use ∞-norm. If False, use 2-norm.
        """
        self.env = env
        self.use_infinity_norm = use_infinity_norm
        
        # Get Lipschitz constant L and time step dt
        L_f, _ = env.get_lipschitz_constants()
        self.L = L_f
        self.dt = env.dt
        
        # Precompute exponential growth factor e^(Lt)
        self.growth_factor = np.exp(self.L * self.dt)
        
        print(f"\nGrönwall Reachability Initialized:")
        print(f"  Lipschitz constant L = {self.L}")
        print(f"  Time step dt = {self.dt}")
        print(f"  Growth factor e^(Lt) = {self.growth_factor:.6f}")
        print(f"  Norm type: {'∞-norm (max)' if use_infinity_norm else '2-norm (Euclidean)'}")
        
        # Compare to simple Lipschitz
        simple_factor = 1 + self.L * self.dt
        print(f"  Simple Lipschitz would use: {simple_factor:.6f}")
        print(f"  Grönwall is {100*(self.growth_factor/simple_factor - 1):.2f}% larger (correct!)")
    
    @staticmethod
    def fix_angle_interval(left_angle: float, right_angle: float) -> Tuple[float, float]:
        """
        Normalize an angle interval to a canonical form.
        
        This function ensures:
        1. right_angle >= left_angle (unwrap if needed)
        2. If interval spans full circle, return [-π, π]
        3. Otherwise, shift interval to be centered around [-π, π]
        
        Based on professor's implementation.
        
        Args:
            left_angle: Lower bound of angle interval
            right_angle: Upper bound of angle interval
            
        Returns:
            Tuple (normalized_left, normalized_right)
        """
        # Handle degenerate case (almost point interval)
        if abs(right_angle - left_angle) < 0.001:
            left_angle = right_angle - 0.01
        
        # Unwrap: ensure right >= left
        while right_angle < left_angle:
            right_angle += 2 * np.pi
        
        # Full circle case
        if right_angle - left_angle >= 2 * np.pi - 0.01:
            return -np.pi, np.pi
        
        # Shift interval to avoid being too far from [-π, π]
        # Move up if both are too negative
        while left_angle < -np.pi or right_angle < -np.pi:
            left_angle += 2 * np.pi
            right_angle += 2 * np.pi
        
        # Move down if both are too positive
        while left_angle > 3 * np.pi or right_angle > 3 * np.pi:
            left_angle -= 2 * np.pi
            right_angle -= 2 * np.pi
        
        # Final shift: if both > π, bring down
        while left_angle > np.pi and right_angle > np.pi:
            left_angle -= 2 * np.pi
            right_angle -= 2 * np.pi
        
        return left_angle, right_angle
    
    @staticmethod
    def intervals_intersect(a_left: float, a_right: float, 
                          b_left: float, b_right: float) -> bool:
        """
        Check if two angle intervals intersect.
        
        IMPORTANT: Assumes both intervals are already normalized via fix_angle_interval.
        This means right >= left for both intervals (possibly with 2π unwrapping).
        
        The intersection logic is now simple:
        - No intersection if: a_right < b_left OR b_right < a_left
        - Otherwise: they intersect
        
        Args:
            a_left, a_right: First interval [a_left, a_right]
            b_left, b_right: Second interval [b_left, b_right]
            
        Returns:
            True if intervals overlap, False otherwise
        """
        # After normalization, intervals don't wrap, so standard interval logic works
        return not (a_right < b_left or b_right < a_left)
    
    def compute_reachable_set(self, cell, action) -> np.ndarray:
        """
        Compute reachable set bounds using r * e^(Lt).
        
        FIXED: Uses fix_angle_interval for theta instead of manual wraparound logic.
        
        Returns:
            reach_bounds: [dim, 2] array of [min, max] for each dimension
        """
        # Step 1: Get cell center
        center = cell.center
        
        # Step 2: Compute cell radius r
        if self.use_infinity_norm:
            r = 0.5 * cell.get_max_range()
        else:
            ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
            r = 0.5 * np.linalg.norm(ranges)
        
        # Step 3: Propagate center forward one time step
        center_next = self.env.dynamics(center, action)
        
        # Step 4: Compute Grönwall expansion: ε = r * e^(Lt)
        expansion = r * self.growth_factor
        
        # Step 5: Create bounding box
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        
        # For x and y (Cartesian coordinates): simple ± expansion
        reach_bounds[0, :] = [center_next[0] - expansion, center_next[0] + expansion]
        reach_bounds[1, :] = [center_next[1] - expansion, center_next[1] + expansion]
        
        # For θ (angular coordinate): use fix_angle_interval
        theta_lower = center_next[2] - expansion
        theta_upper = center_next[2] + expansion
        
        # Normalize the angle interval
        theta_lower, theta_upper = self.fix_angle_interval(theta_lower, theta_upper)
        reach_bounds[2, :] = [theta_lower, theta_upper]
        
        return reach_bounds
    
    def compute_successor_cells(self, cell, action, cell_tree) -> List[Cell]:
        """
        Find all cells that intersect the reachable set.
        
        FIXED: Simplified theta intersection using normalized intervals.
        No more wraparound detection needed!
        
        Returns:
            List of successor cells
        """
        # Step 1: Get reachable bounds (already normalized)
        reach_bounds = self.compute_reachable_set(cell, action)
        
        successors = []
        
        # Step 2: Check each leaf cell for intersection
        for candidate in cell_tree.get_leaves():
            # Step 2a: Check x-coordinate overlap (standard interval)
            if (candidate.bounds[0, 1] < reach_bounds[0, 0] or 
                candidate.bounds[0, 0] > reach_bounds[0, 1]):
                continue
            
            # Step 2b: Check y-coordinate overlap (standard interval)
            if (candidate.bounds[1, 1] < reach_bounds[1, 0] or 
                candidate.bounds[1, 0] > reach_bounds[1, 1]):
                continue
            
            # Step 2c: Check θ-coordinate overlap using normalized intervals
            # Normalize both intervals first
            cand_theta_left, cand_theta_right = self.fix_angle_interval(
                candidate.bounds[2, 0], 
                candidate.bounds[2, 1]
            )
            reach_theta_left = reach_bounds[2, 0]
            reach_theta_right = reach_bounds[2, 1]
            
            # Check intersection using normalized intervals
            if not self.intervals_intersect(
                cand_theta_left, cand_theta_right,
                reach_theta_left, reach_theta_right
            ):
                continue
            
            # Step 3: Add to successors if all dimensions overlap
            successors.append(candidate)
        
        return successors
# ============================================================================
# PART 4: VALUE ITERATION (ALGORITHM 1 - FIXED)
# ============================================================================

class SafetyValueIterator:
    """
    Implements Algorithm 1 with fixes:
    - Practical convergence: run full VI until convergence
    - Proper initialization and plotting
    """
    
    def __init__(self, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: GronwallReachabilityAnalyzer, output_dir: Optional[str] = None):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        
        if output_dir is None:
            rname = type(reachability).__name__
            output_dir = f"./results/{rname}"
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.L_f, self.L_l = env.get_lipschitz_constants()
        if gamma * self.L_f >= 1:
            raise ValueError(f"Contraction condition violated: γL_f = {gamma * self.L_f} >= 1")
        
        self.refinement_phase = 0
        
        print(f"Initialized with γ={gamma}, L_f={self.L_f}, L_l={self.L_l}")
        print(f"Contraction factor: γL_f = {gamma * self.L_f:.4f}")
    
    def initialize_cells(self):
        """Initialize stage cost bounds for all cells."""
        for cell in self.cell_tree.get_leaves():
            l_center = self.env.failure_function(cell.center)
            r = 0.5 * cell.get_max_range()#eta/2
            cell.l_lower = l_center - self.L_l * r
            cell.l_upper = l_center + self.L_l * r
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
        
    def initialize_new_cells(self, new_cells: List[Cell]):
        """
        Initialize new cells after refinement.
        
        Always initializes with V_lower = -∞ and V_upper = +∞ (no parent inheritance).
        The local value iteration will then compute proper bounds.
        """
        for cell in new_cells:
            # Compute stage cost bounds
            l_center = self.env.failure_function(cell.center)
            r = 0.5 * cell.get_max_range()#eta/2
            cell.l_lower = l_center - self.L_l * r
            cell.l_upper = l_center + self.L_l * r
            
            # Always initialize value bounds to -∞/+∞ (conservative initialization)
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
    # def bellman_update(self, cell: Cell) -> Tuple[float, float]:
    #     """Single Bellman update for a cell."""
    #     max_upper = -np.inf###CHECK THIS
    #     max_lower = -np.inf
        
    #     for action in self.env.get_action_space():
    #         successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
    #         if len(successors) == 0:
    #             continue
            
    #         upper_vals = [s.V_upper for s in successors if s.V_upper is not None]
    #         if upper_vals:
    #             action_upper = self.gamma * max(upper_vals)
    #             max_upper = max(max_upper, action_upper)
            
    #         lower_vals = [s.V_lower for s in successors if s.V_lower is not None]
    #         if lower_vals:
    #             action_lower = self.gamma * min(lower_vals)
    #             max_lower = max(max_lower, action_lower)
        
    #     new_V_upper = min(cell.l_upper, max_upper) if max_upper > -np.inf else cell.l_upper
    #     new_V_lower = min(cell.l_lower, max_lower) if max_lower > -np.inf else cell.l_lower
    #     return new_V_upper, new_V_lower
    def bellman_update(self, cell: Cell) -> Tuple[float, float]:
        """
        Single Bellman update for a cell.
        
        CONSERVATIVE VERSION: Uses the same action for both upper and lower bounds.
        The action is selected to maximize the lower bound, and that same action's
        upper bound is used for the upper value update.
        """
        best_min_val = -np.inf
        best_max_val = -np.inf
        
        for action in self.env.get_action_space():
            successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
            if len(successors) == 0:
                continue
            
            # Compute BOTH bounds for this action
            upper_vals = [s.V_upper for s in successors if s.V_upper is not None]
            lower_vals = [s.V_lower for s in successors if s.V_lower is not None]
            
            if not lower_vals or not upper_vals:
                continue
            
            # For THIS action: min over lower bounds, max over upper bounds
            action_lower = self.gamma * min(lower_vals)
            action_upper = self.gamma * max(upper_vals)
            
            # Select action that maximizes the lower bound (conservative choice)
            if action_lower > best_min_val:
                best_min_val = action_lower
                best_max_val = action_upper  # Use the SAME action's upper bound
        
        # Apply the min with stage costs
        new_V_lower = min(cell.l_lower, best_min_val) if best_min_val > -np.inf else cell.l_lower
        new_V_upper = min(cell.l_upper, best_max_val) if best_max_val > -np.inf else cell.l_upper
        
        return new_V_upper, new_V_lower
    
    def value_iteration(self, max_iterations: int = 1000, convergence_tol: float = 1e-3,
                       plot_freq: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run value iteration until convergence.
        
        Practical interpretation: Run VI on all cells until overall convergence,
        then identify boundary cells for refinement.
        """
        self.initialize_cells()
        conv_history_upper = []
        conv_history_lower = []
        
        print(f"\nStarting value iteration (max {max_iterations} iterations)...")
        print(f"Convergence tolerance: {convergence_tol}")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        
        for iteration in range(max_iterations):
            prev_upper = {cell.cell_id: cell.V_upper for cell in self.cell_tree.get_leaves()}
            prev_lower = {cell.cell_id: cell.V_lower for cell in self.cell_tree.get_leaves()}
            
            updates = {}
            for cell in self.cell_tree.get_leaves():
                new_upper, new_lower = self.bellman_update(cell)
                updates[cell.cell_id] = (new_upper, new_lower)
                cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            # for cell in self.cell_tree.get_leaves():
            #     cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            diff_upper = max(abs(cell.V_upper - prev_upper[cell.cell_id]) 
                           for cell in self.cell_tree.get_leaves())
            diff_lower = max(abs(cell.V_lower - prev_lower[cell.cell_id]) 
                           for cell in self.cell_tree.get_leaves())
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"Iteration {iteration + 1}: "
                  f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.6f}, "
                  f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.6f}")
            
            if (iteration + 1) % plot_freq == 0:
                self._save_plot(iteration + 1)
            
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"\n✓ Converged at iteration {iteration + 1}!")
                self._save_plot(iteration + 1, final=True)
                break
        else:
            print(f"\nReached maximum iterations ({max_iterations})")
            self._save_plot(max_iterations, final=True)
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
    
    def local_value_iteration(self, updated_cells: Set[Cell], max_iterations: int = 100,
                            convergence_tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        """
        LOCAL value iteration: only update BOUNDARY cells.
        
        After refinement, we identify all cells where V̄(s) > 0 and V_(s) < 0
        (the boundary/uncertain region) and run value iteration only on those cells.
        
        Safe cells (V_(s) > 0) and unsafe cells (V̄(s) < 0) are NOT updated,
        even if their successor sets include refined cells.
        
        Args:
            updated_cells: The newly created cells from refinement (used for tracking only)
            max_iterations: Maximum number of iterations
            convergence_tol: Convergence tolerance
            
        Returns:
            Tuple of convergence histories (upper, lower)
        """
        # Identify ALL boundary cells in the current grid
        boundary_cells = set()
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                if cell.V_upper > 0 and cell.V_lower < 0:
                    boundary_cells.add(cell)
        
        print(f"  Local VI: {len(boundary_cells)} boundary cells "
            f"(out of {self.cell_tree.get_num_leaves()} total)")
        print(f"    Including {len(updated_cells & boundary_cells)} newly created cells")
        
        if len(boundary_cells) == 0:
            print("    No boundary cells to update!")
            return np.array([]), np.array([])
        
        conv_history_upper = []
        conv_history_lower = []
        
        for iteration in range(max_iterations):
            prev_upper = {cell.cell_id: cell.V_upper for cell in boundary_cells}
            prev_lower = {cell.cell_id: cell.V_lower for cell in boundary_cells}
            
            updates = {}
            for cell in boundary_cells:
                new_upper, new_lower = self.bellman_update(cell)
                updates[cell.cell_id] = (new_upper, new_lower)
            
            for cell in boundary_cells:
                cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            diff_upper = max(abs(cell.V_upper - prev_upper[cell.cell_id]) 
                            for cell in boundary_cells)
            diff_lower = max(abs(cell.V_lower - prev_lower[cell.cell_id]) 
                            for cell in boundary_cells)
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"    Iter {iteration + 1}: "
                f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.6f}, "
                f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.6f}")
            
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"    ✓ Local VI converged at iteration {iteration + 1}!")
                break
        
        return np.array(conv_history_upper), np.array(conv_history_lower)


    def _identify_boundary_cells(self) -> List[Cell]:
        """Identify boundary cells where V̄_γ(s) > 0 and V_γ(s) < 0."""
        boundary = []
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                if cell.V_upper > 0 and cell.V_lower < 0:
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
# PART 5: ADAPTIVE REFINEMENT (ALGORITHM 2/3 - FIXED)
# ============================================================================

class AdaptiveRefinement:
    """
    Algorithm 2/3 with fixes:
    - LOCAL value iteration after refinement
    - Proper plot generation after each phase
    - Detailed logging
    """
    
    def __init__(self,args, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: GronwallReachabilityAnalyzer, output_dir: str = None):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.args=args
        
        if output_dir is None:
            rname = type(reachability).__name__
            output_dir = f"./results_adaptive/{rname}"
        self.output_dir = output_dir
        
        self.value_iterator = SafetyValueIterator(env, gamma, cell_tree, reachability, output_dir)
        self.L_l = env.get_lipschitz_constants()[1]
    
    def refine(self, epsilon: float, max_refinements: int = 100,
               vi_iterations_per_refinement: int = 100):
        """Main adaptive refinement loop with proper LOCAL VI."""
        eta_min = epsilon / (self.L_l)
        
        print(f"\n{'='*70}")
        print("ADAPTIVE REFINEMENT CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Error tolerance ε: {epsilon}")
        print(f"  Minimum cell size η_min: {eta_min:.6f}")
        print(f"  Max refinements: {max_refinements}")
        print(f"  VI iterations per refinement: {vi_iterations_per_refinement}")
        
        # Phase 0: Initial value iteration on full grid
        print(f"\n{'='*70}")
        print("PHASE 0: INITIAL VALUE ITERATION")
        print(f"{'='*70}")
        print(f"Grid: {self.cell_tree.get_num_leaves()} cells")
        
        self.value_iterator.refinement_phase = 0
        self.value_iterator.value_iteration(
            max_iterations=vi_iterations_per_refinement,
            plot_freq=self.args.plot_freq
        )
        
        # Save plot after initial VI
        filename = os.path.join(self.output_dir, "value_function_phase_0_complete.png")
        plot_value_function(self.env, self.cell_tree, filename, 0)
        
        # Plot initial l bounds
        plot_failure_function_bounds(
            self.env, self.cell_tree,
            filename_prefix="ell_bounds_phase_0",
            save_dir=self.output_dir
        )
        
        # Initial queue state
        boundary_cells = self._identify_boundary_cells()
        refinable = [c for c in boundary_cells if c.get_max_range() > eta_min]
        
        print(f"\n{'='*70}")
        print("INITIAL QUEUE STATE")
        print(f"{'='*70}")
        print(f"Boundary cells in queue: {len(boundary_cells)}")
        print(f"  Refinable (>η_min): {len(refinable)}")
        print(f"  Below threshold: {len(boundary_cells) - len(refinable)}")
        
        # Refinement loop
        refinement_iter = 0
        total_refined = 0
        
        while refinement_iter < max_refinements:
            boundary_cells = self._identify_boundary_cells()
            refinable = [c for c in boundary_cells if c.get_max_range() > eta_min]
            too_small = [c for c in boundary_cells if c.get_max_range() <= eta_min]
            
            print(f"\n{'='*70}")
            print(f"PHASE {refinement_iter + 1}: REFINEMENT")
            print(f"{'='*70}")
            print(f"Queue size: {len(boundary_cells)} boundary cells")
            print(f"  Refinable (>η_min): {len(refinable)}")
            print(f"  Below threshold (<η_min): {len(too_small)}")
            
            if len(refinable) == 0:
                print(f"\n✓ STOPPING: All {len(too_small)} boundary cells below η_min")
                break
            
            # Cell size statistics
            if refinable:
                sizes = [c.get_max_range() for c in refinable]
                print(f"\nRefinable cell size statistics:")
                print(f"  Max: {max(sizes):.6f}")
                print(f"  Mean: {np.mean(sizes):.6f}")
                print(f"  Min: {min(sizes):.6f}")
                print(f"  Threshold η_min: {eta_min:.6f}")
            
            # Perform refinement
            print(f"\nRefining {len(refinable)} cells...")
            new_cells = []
            for cell in refinable:
                self.cell_tree.refine_cell(cell)
                new_cells.extend(cell.children)
            
            total_refined += len(refinable)
            print(f"  Refined: {len(refinable)} parent cells")
            print(f"  Created: {len(new_cells)} child cells")
            print(f"  Total cells now: {self.cell_tree.get_num_leaves()}")
            print(f"  Cumulative refined: {total_refined}")
            
            # Initialize new cells with parent value inheritance or with inf / -inf
            if new_cells:
                print(f"\nInitializing {len(new_cells)} new cells...")
                self.value_iterator.initialize_new_cells(new_cells)
                
                # inherited = sum(1 for c in new_cells 
                #               if c.V_upper != np.inf and c.V_lower != -np.inf)
                # print(f"  Inherited parent values: {inherited}/{len(new_cells)} cells")
                
                # Plot l bounds after refinement
                plot_failure_function_bounds(
                    self.env, self.cell_tree,
                    filename_prefix=f"ell_bounds_phase_{refinement_iter + 1}",
                    save_dir=self.output_dir
                )
                
                
            
            # Set refinement phase BEFORE LOCAL VI
            self.value_iterator.refinement_phase = refinement_iter + 1
            
            # LOCAL value iteration (FIXED: only on affected cells)
            print(f"\nLocal Value Iteration (Phase {refinement_iter + 1}):")
            
            conv_upper, conv_lower = self.value_iterator.local_value_iteration(
                updated_cells=set(new_cells),
                max_iterations=vi_iterations_per_refinement,
                convergence_tol=self.args.tolerance
            )
            # plot_value_function(self.env, self.cell_tree, filename, refinement_iter + 1)
            
            # Save plot AFTER local VI completes
            filename = os.path.join(
                self.output_dir, 
                f"value_function_phase_{refinement_iter + 1}_complete.png"
            )
            plot_value_function(self.env, self.cell_tree, filename, refinement_iter + 1)#moved up next to plot ell_bounds
            
            # Post-VI queue state
            # new_boundary = self._identify_boundary_cells()
            # new_refinable = [c for c in new_boundary if c.get_max_range() > eta_min]
            
            new_boundary = self._identify_boundary_cells()
            new_refinable = [c for c in boundary_cells if c.get_max_range() > eta_min]
            too_small = [c for c in boundary_cells if c.get_max_range() <= eta_min]
            print(f"\n  Post-VI Queue State:")
            print(f"    Total boundary: {len(new_boundary)}")
            print(f"    Refinable: {len(new_refinable)}")
            print(f"    Change: {len(new_boundary) - len(boundary_cells):+d} boundary cells")
            
            refinement_iter += 1
        
        # Final summary
        print(f"\n{'='*70}")
        print("ADAPTIVE REFINEMENT COMPLETE")
        print(f"{'='*70}")
        print(f"\nRefinement Summary:")
        print(f"  Phases completed: {refinement_iter}")
        print(f"  Parent cells refined: {total_refined}")
        print(f"  Child cells created: {total_refined * 2}")
        print(f"  Final cells: {self.cell_tree.get_num_leaves()}")
        
        # Final queue state
        final_boundary = self._identify_boundary_cells()
        final_refinable = [c for c in final_boundary if c.get_max_range() > eta_min]
        
        print(f"\nFinal Queue State:")
        print(f"  Boundary cells: {len(final_boundary)}")
        print(f"  Refinable: {len(final_refinable)}")
        print(f"  Below threshold: {len(final_boundary) - len(final_refinable)}")
        
        self._print_statistics()
        
        # Final plots
        plot_failure_function_bounds(
            self.env, self.cell_tree,
            filename_prefix="ell_bounds_FINAL",
            save_dir=self.output_dir
        )
        
        filename = os.path.join(self.output_dir, "value_function_FINAL.png")
        # plot_value_function(self.env, self.cell_tree, filename, 9999)
        
        print(f"\n✓ All results saved to: {self.output_dir}/")
    
    def _identify_boundary_cells(self) -> List[Cell]:
        """Identify boundary cells where V̄_γ(s) > 0 and V_γ(s) < 0."""
        boundary = []
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                if cell.V_upper > 0 and cell.V_lower < 0:
                    boundary.append(cell)
        return boundary
    
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
# PART 6: VISUALIZATION (FIXED)
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
    
    FIXED: Uses tree traversal to find correct leaf cell for each point.
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
    Plot value function by directly coloring leaf cells.
    
    FIXED: No more 80x80 sampling grid - directly use leaf cell values.
    Only plots cells whose theta range includes the current slice.
    """
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Collect values from all relevant cells to determine color scale
    values = []
    relevant_cells = []
    
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
    vmin = min(values)
    vmax = max(values)
    
    # Use a diverging colormap centered at 0 (safe/unsafe boundary)
    from matplotlib.colors import TwoSlopeNorm, Normalize
    cmap = plt.cm.RdYlGn
    
    # Handle edge cases for TwoSlopeNorm
    # TwoSlopeNorm requires: vmin < vcenter < vmax (strict inequalities)
    epsilon = 1e-6  # Small value to ensure strict inequalities
    
    if abs(vmax - vmin) < 1e-10:
        # All values are essentially the same
        if abs(vmin) < 1e-10:
            # All values are ~0
            norm = Normalize(vmin=-0.1, vmax=0.1)
        else:
            # All values are the same non-zero value
            norm = Normalize(vmin=vmin-0.1*abs(vmin), vmax=vmax+0.1*abs(vmax))
    elif vmin > 0:
        # All values are strictly positive
        norm = TwoSlopeNorm(vmin=0, vcenter=max(0, vmin-epsilon), vmax=vmax)
    elif vmax < 0:
        # All values are strictly negative
        norm = TwoSlopeNorm(vmin=vmin, vcenter=min(0, vmax+epsilon), vmax=0)
    elif vmin == 0 and vmax > 0:
        # Values in [0, vmax]
        norm = TwoSlopeNorm(vmin=0, vcenter=epsilon, vmax=vmax)
    elif vmin < 0 and vmax == 0:
        # Values in [vmin, 0]
        norm = TwoSlopeNorm(vmin=vmin, vcenter=-epsilon, vmax=0)
    elif vmin == 0 and vmax == 0:
        # All values are exactly 0
        norm = Normalize(vmin=-0.1, vmax=0.1)
    else:
        # Values span zero properly (vmin < 0 < vmax)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    
    # Plot each cell as a colored rectangle
    for cell, value in relevant_cells:
        a_x, b_x = cell.bounds[0]
        a_y, b_y = cell.bounds[1]
        
        color = cmap(norm(value))
        
        rect = Rectangle(
            (a_x, a_y),
            b_x - a_x,
            b_y - a_y,
            facecolor=color,
            edgecolor='black',
            linewidth=0.3,
            alpha=0.9
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            color='red',
            alpha=0.6,
            edgecolor='darkred',
            linewidth=2,
            zorder=15
        )
        ax.add_patch(obstacle)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)
    
    # Add colorbar with explicit ticks
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Create colorbar
    cbar = plt.colorbar(sm, ax=ax, label=label)
    
    # Set explicit tick locations to ensure labels appear
    # Always include min, 0, and max
    if vmin < 0 and vmax > 0:
        # Spanning zero
        tick_vals = [vmin, vmin/2, 0, vmax/2, vmax]
    elif vmin >= 0:
        # All non-negative
        tick_vals = [0, vmax/3, 2*vmax/3, vmax]
    elif vmax <= 0:
        # All non-positive
        tick_vals = [vmin, 2*vmin/3, vmin/3, 0]
    else:
        # Fallback
        tick_vals = [vmin, 0, vmax]
    
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v:.3f}' for v in tick_vals])
def _plot_classification_slice(
    env: Environment,
    cell_tree: CellTree,
    theta: float,
    ax
):
    """
    Plot cell classification by directly coloring leaf cells.
    
    FIXED: No more 80x80 sampling grid - directly use leaf cells.
    """
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    # Colors for classification
    C_UNSAFE = "#d62728"  # Red
    C_SAFE = "#2ca02c"    # Green
    C_BOUND = "#7f7f7f"   # Gray
    C_UNKNOWN = "#ffffff"  # White (for cells without values)
    
    # Plot each cell directly
    for cell in cell_tree.get_leaves():
        # Only plot cells whose theta range includes this slice
        theta_min, theta_max = cell.bounds[2]
        if not (theta_min <= theta <= theta_max):
            continue
        
        # Determine cell color based on classification
        if cell.V_upper is None or cell.V_lower is None:
            color = C_UNKNOWN
        elif cell.V_lower > 0:
            color = C_SAFE
        elif cell.V_upper < 0:
            color = C_UNSAFE
        else:
            color = C_BOUND
        
        # Draw colored rectangle
        rect = Rectangle(
            (cell.bounds[0, 0], cell.bounds[1, 0]),
            cell.get_range(0),
            cell.get_range(1),
            facecolor=color,
            edgecolor='black',
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
            alpha=0.6,
            edgecolor='darkred',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(obstacle)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(False)
    
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
    """Plots convergence history showing geometric contraction."""
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
    """Visualizes reachable cells from a single source cell for each action."""
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    n_cells = len(leaves)
    if cell_idx is None:
        cell_idx = n_cells // 2
    
    src_cell = leaves[cell_idx]
    src_center = src_cell.center[:2]
    theta_center = src_cell.center[2]
    
    bounds = env.get_state_bounds()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_aspect('equal')
    # ax.grid(True, alpha=0.2)
    ax.set_title(
        f"Reachable cells from cell #{cell_idx} "
        f"(θ={theta_center:.2f} rad, {np.degrees(theta_center):.1f}°)"
    )
    
    action_colors = {-1.0: 'blue', 0.0: 'green', 1.0: 'orange'}
    offset_angles = {-1.0: -8*np.pi/180, 0.0: 0.0, 1.0: 8*np.pi/180}
    
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
    
    for action in env.get_action_space():
        succ_cells = reachability.compute_successor_cells(src_cell, action, cell_tree)
        color = action_colors[action]
        offset_angle = offset_angles[action]
        
        for dst_cell in succ_cells:
            dst_center = dst_cell.center[:2]
            dx = dst_center[0] - src_center[0]
            dy = dst_center[1] - src_center[1]
            
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


def plot_reachability_multiple_cells(
    env,
    cell_tree,
    reachability,
    num_cells=5,
    cell_indices=None,
    n_samples=10,
    save_dir="./results",
    filename_prefix="reachability"
):
    """
    Visualizes reachable cells from multiple source cells.
    
    Args:
        num_cells: Number of random cells to plot (ignored if cell_indices provided)
        cell_indices: Specific list of cell indices to plot. If None, randomly sample.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    n_cells = len(leaves)
    
    # Determine which cells to plot
    if cell_indices is None:
        # Sample random cells (avoid duplicates)
        rng = np.random.default_rng(42)
        cell_indices = rng.choice(n_cells, size=min(num_cells, n_cells), replace=False)
    
    print(f"\nGenerating reachability plots for {len(cell_indices)} cells...")
    
    # Plot each cell
    for idx in cell_indices:
        plot_reachability_single_cell(
            env=env,
            cell_tree=cell_tree,
            reachability=reachability,
            cell_idx=int(idx),
            n_samples=n_samples,
            save_dir=save_dir,
            filename_prefix=filename_prefix
        )


def plot_reachability_grid(
    env,
    cell_tree,
    reachability,
    grid_size=(2, 3),
    cell_indices=None,
    n_samples=10,
    figsize=(15, 10),
    save_dir="./results",
    filename="reachability_grid.png"
):
    """
    Visualizes reachable cells from multiple source cells in a single figure grid.
    
    Args:
        grid_size: Tuple (rows, cols) for subplot layout
        cell_indices: List of cell indices to plot. If None, sample randomly.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    n_cells = len(leaves)
    num_plots = grid_size[0] * grid_size[1]
    
    # Determine which cells to plot
    if cell_indices is None:
        rng = np.random.default_rng(42)
        cell_indices = rng.choice(n_cells, size=min(num_plots, n_cells), replace=False)
    
    cell_indices = cell_indices[:num_plots]  # Trim to fit grid
    
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    action_colors = {-1.0: 'blue', 0.0: 'green', 1.0: 'orange'}
    offset_angles = {-1.0: -8*np.pi/180, 0.0: 0.0, 1.0: 8*np.pi/180}
    bounds = env.get_state_bounds()
    
    print(f"\nGenerating grid reachability plot for {len(cell_indices)} cells...")
    
    for plot_idx, cell_idx in enumerate(cell_indices):
        ax = axes[plot_idx]
        src_cell = leaves[cell_idx]
        src_center = src_cell.center[:2]
        theta_center = src_cell.center[2]
        
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.set_aspect('equal')
        # ax.grid(True, alpha=0.2)
        ax.set_title(
            f"Cell #{cell_idx} (θ={np.degrees(theta_center):.0f}°)",
            fontsize=10
        )
        
        # Draw all cells
        for cell in leaves:
            rect = Rectangle(
                (cell.bounds[0, 0], cell.bounds[1, 0]),
                cell.get_range(0),
                cell.get_range(1),
                fill=False,
                edgecolor='gray',
                linewidth=0.3,
                alpha=0.2
            )
            ax.add_patch(rect)
        
        # Highlight source cell
        rect_src = Rectangle(
            (src_cell.bounds[0, 0], src_cell.bounds[1, 0]),
            src_cell.get_range(0),
            src_cell.get_range(1),
            fill=False,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(rect_src)
        ax.plot(*src_center, 'ko', markersize=4)
        
        # Draw reachability arrows
        for action in env.get_action_space():
            succ_cells = reachability.compute_successor_cells(src_cell, action, cell_tree)
            color = action_colors[action]
            offset_angle = offset_angles[action]
            
            for dst_cell in succ_cells:
                dst_center = dst_cell.center[:2]
                dx = dst_center[0] - src_center[0]
                dy = dst_center[1] - src_center[1]
                
                if offset_angle != 0.0:
                    rot = np.array([
                        [np.cos(offset_angle), -np.sin(offset_angle)],
                        [np.sin(offset_angle), np.cos(offset_angle)]
                    ])
                    dx, dy = rot @ np.array([dx, dy])
                
                ax.arrow(src_center[0], src_center[1], dx, dy,
                        color=color, alpha=0.5, linewidth=1.5,
                        head_width=0.08, length_includes_head=True)
        
        # Draw obstacle
        if isinstance(env, DubinsCarEnvironment):
            obstacle = Circle(
                env.obstacle_position,
                env.obstacle_radius,
                color='red',
                alpha=0.4,
                zorder=4
            )
            ax.add_patch(obstacle)
        
        ax.set_xlabel('x', fontsize=8)
        ax.set_ylabel('y', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # Hide unused subplots
    for idx in range(len(cell_indices), len(axes)):
        axes[idx].axis('off')
    
    # Add global legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Action -1'),
        plt.Line2D([0], [0], color='green', linewidth=2, label='Action 0'),
        plt.Line2D([0], [0], color='orange', linewidth=2, label='Action +1')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"Saved grid reachability plot to {filepath}")
def plot_failure_function_bounds(
    env,
    cell_tree,
    filename_prefix="ell_bounds",
    save_dir="./results"
):
    """Visualizes l_lower and l_upper for all cells as two side-by-side heatmaps."""
    os.makedirs(save_dir, exist_ok=True)
    
    leaves = cell_tree.get_leaves()
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    
    vals_lower = [c.l_lower for c in leaves if c.l_lower is not None]
    vals_upper = [c.l_upper for c in leaves if c.l_upper is not None]
    
    if not vals_lower or not vals_upper:
        return
    
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
            
            cx, cy = cell.center[:2]
            color_text = "white" if val < 0 else "black"
            ax.text(cx, cy, f"{val:.2f}", ha="center", va="center",
                   fontsize=6, color=color_text)
        
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
        # ax.grid(True, alpha=0.3)
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(r"$\ell$ value", rotation=270, labelpad=15)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ℓ-bounds plot to {filename}")

# Add this to PART 6: VISUALIZATION section, after the other plotting functions

def compute_ground_truth_reachability(
    env: Environment,
    time_horizon: int = 100,
    resolution: int = 20
):
    """
    Compute ground truth safe set by forward simulation with a hand-designed controller.
    
    Simply samples (x, y, theta) on a grid, simulates forward with safe controller,
    and checks if trajectory hits the failure set.
    """
    print(f"\n{'='*70}")
    print("COMPUTING GROUND TRUTH REACHABILITY")
    print(f"{'='*70}")
    print(f"Time horizon: {time_horizon} steps")
    
    # Only compute for theta slices we'll actually plot
    theta_slices = [0, np.pi/4, np.pi/2]
    print(f"Grid resolution: {resolution}^2 x {len(theta_slices)} slices = {resolution**2 * len(theta_slices)} states")
    
    def safe_controller(state: np.ndarray, env: DubinsCarEnvironment) -> float:
        """
        Hand-designed safe controller: actively avoid obstacle.
        """
        pos = state[:2]
        theta = state[2]
        
        # SAFE MODE: Move away from obstacle
        away_from_obstacle = pos - env.obstacle_position
        desired_heading = np.arctan2(away_from_obstacle[1], away_from_obstacle[0])
        
        angle_diff = np.arctan2(
            np.sin(desired_heading - theta),
            np.cos(desired_heading - theta)
        )
        
        actions = env.get_action_space()
        if angle_diff > 0:
            return actions[-1]  # Turn left
        else:
            return actions[0]   # Turn right
     

    def simulate_trajectory(initial_state: np.ndarray, time_horizon: int) -> bool:
        """
        Simulate trajectory with safe controller.
        Returns True if safe (never enters failure set), False otherwise.
        """
        state = initial_state.copy()
        
        for t in range(time_horizon):
            # Check failure
            failure_value = env.failure_function(state)
            if failure_value < 0:
                return False  # Entered failure set
            
            # Apply safe controller
            action = safe_controller(state, env)
            state = env.dynamics(state, action)
            
            # Check bounds
            bounds = env.get_state_bounds()
            for j in range(len(state)):
                state[j] = np.clip(state[j], bounds[j, 0], bounds[j, 1])
        
        return True  # Stayed safe
    
    # Create grid of states to test
    bounds = env.get_state_bounds()
    x_vals = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y_vals = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    
    # Store results: (x, y, theta) -> is_safe
    results = {}
    
    print("\nSimulating trajectories...")
    total = resolution ** 2 * len(theta_slices)
    count = 0
    
    # Only loop over the theta slices we care about
    for theta in theta_slices:
        print(f"\n  Computing for θ = {theta:.2f} rad ({np.degrees(theta):.0f}°)...")
        for x in x_vals:
            for y in y_vals:
                state = np.array([x, y, theta])
                is_safe = simulate_trajectory(state, time_horizon)
                results[(x, y, theta)] = is_safe
                
                count += 1
                if count % 500 == 0:
                    print(f"    Progress: {count}/{total} states ({100*count/total:.1f}%)")
    
    # Count statistics
    safe_count = sum(1 for v in results.values() if v)
    unsafe_count = len(results) - safe_count
    
    print(f"\n{'='*70}")
    print("GROUND TRUTH CLASSIFICATION")
    print(f"{'='*70}")
    print(f"Safe states:   {safe_count:6d} ({100*safe_count/total:5.1f}%)")
    print(f"Unsafe states: {unsafe_count:6d} ({100*unsafe_count/total:5.1f}%)")
    
    # Plot ground truth at different theta slices
    fig, axes = plt.subplots(1, len(theta_slices), figsize=(5*len(theta_slices), 5))
    
    if len(theta_slices) == 1:
        axes = [axes]
    
    for idx, (ax, theta_slice) in enumerate(zip(axes, theta_slices)):
        # Create 2D grid for this theta slice
        safety_grid = np.zeros((resolution, resolution))
        
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                is_safe = results[(x, y, theta_slice)]
                safety_grid[i, j] = 1.0 if is_safe else 0.0
        
        # Plot
        im = ax.imshow(
            safety_grid,
            extent=[bounds[0, 0], bounds[0, 1], bounds[1, 0], bounds[1, 1]],
            origin='lower',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            aspect='auto',
            alpha=0.9
        )
        
        # Draw obstacle
        if isinstance(env, DubinsCarEnvironment):
            obstacle = Circle(
                env.obstacle_position,
                env.obstacle_radius,
                color='red',
                alpha=0.6,
                edgecolor='black',
                linewidth=2,
                zorder=10
            )
            ax.add_patch(obstacle)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(bounds[0, 0], bounds[0, 1])
        ax.set_ylim(bounds[1, 0], bounds[1, 1])
        ax.set_aspect('equal')
        ax.set_title(f'Ground Truth (θ={theta_slice:.2f} rad, {np.degrees(theta_slice):.0f}°)')
        # ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Safe (1) / Unsafe (0)')
    
    fig.suptitle(f'Ground Truth Safe Set (Forward Simulation, T={time_horizon})', 
                 fontsize=14, y=0.98)
    plt.tight_layout()
    
    filepath = "./ground_truth_reachability.png"
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Ground truth plot saved to: {filepath}")
    
    return results
# ============================================================================
# PART 7: MAIN INTERFACE
# ============================================================================

def run_algorithm_1(args):
    """Runs Algorithm 1: Basic discretization with value iteration."""
    print("="*70)
    print("ALGORITHM 1: Discretization Routine")
    print("="*70)
    
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
    
    print(f"\nInitializing grid with resolution {args.resolution}^3...")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.resolution)
    print(f"  Total cells: {cell_tree.get_num_leaves()}")
    
    # SELECT REACHABILITY ANALYZER (add Grönwall option)
    if args.gronwall_reach:
        reachability = GronwallReachabilityAnalyzer(
            env, 
            use_infinity_norm=args.use_inf_norm
        )
        print("  Using Grönwall-based reachability (exponential growth via e^(Lt))")
    elif args.center_only:
        reachability = CenterOnlyReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print("  Using CENTER-ONLY reachability (debugging mode)")
    elif args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print("  Using Lipschitz-based reachability (linear growth)")
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using sampling-based reachability ({args.samples} samples/dim)")
    
    value_iter = SafetyValueIterator(
        env=env,
        gamma=args.gamma,
        cell_tree=cell_tree,
        reachability=reachability,
        output_dir=None
    )
    
    if args.plot_reachability:
        print("\nGenerating reachability visualizations...")
        
        # Option 1: Plot multiple cells as separate files
        plot_reachability_multiple_cells(
            env, cell_tree, reachability,
            num_cells=5,  # Plot 5 random cells
            save_dir=value_iter.output_dir
        )
        
        # Option 2: Plot multiple cells in a single grid figure
        plot_reachability_grid(
            env, cell_tree, reachability,
            grid_size=(2, 3),  # 2 rows, 3 columns = 6 cells
            save_dir=value_iter.output_dir
        )
        
        # Option 3: Plot specific cells (e.g., near obstacle, far from obstacle, etc.)
        leaves = cell_tree.get_leaves()
        
        # Select interesting cells: near obstacle, middle, corners
        interesting_indices = [
            len(leaves) // 4,      # Quarter through
            len(leaves) // 2,      # Middle
            3 * len(leaves) // 4,  # Three quarters
            0,                     # First cell
            len(leaves) - 1        # Last cell
        ]
        
        plot_reachability_multiple_cells(
            env, cell_tree, reachability,
            cell_indices=interesting_indices,
            save_dir=value_iter.output_dir,
            filename_prefix="reachability_interesting"
        )
    
    if args.plot_failure:
        print("Generating failure function bounds visualization...")
        value_iter.initialize_cells()
        plot_failure_function_bounds(
            env, value_iter.cell_tree, 
            save_dir=value_iter.output_dir
        )
    
    start_time = time.time()
    conv_upper, conv_lower = value_iter.value_iteration(
        max_iterations=args.iterations,
        convergence_tol=args.tolerance,
        plot_freq=args.plot_freq
    )
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print("ALGORITHM 1 COMPLETE")
    print("="*70)
    print(f"Total computation time: {elapsed:.2f} seconds")
    print(f"Time per iteration: {elapsed/len(conv_upper):.3f} seconds")
    print(f"Total iterations: {len(conv_upper)}")
    
    plot_convergence(conv_upper, conv_lower, f"{value_iter.output_dir}/convergence.png")
    print(f"\nAll results saved to: {value_iter.output_dir}/")


def run_algorithm_2(args):
    """Runs Algorithm 2/3: Adaptive refinement."""
    print("="*70)
    print("ALGORITHM 2/3: Adaptive Refinement")
    print("="*70)
    
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
    
    print(f"\nInitializing coarse grid with resolution {args.initial_resolution}^3...")
    cell_tree = CellTree(env.get_state_bounds(), initial_resolution=args.initial_resolution)
    print(f"  Initial cells: {cell_tree.get_num_leaves()}")
    
    # SELECT REACHABILITY ANALYZER (add Grönwall option)
    if args.gronwall_reach:
        reachability = GronwallReachabilityAnalyzer(
            env, 
            use_infinity_norm=args.use_inf_norm
        )
        print("  Using Grönwall-based reachability (exponential growth via e^(Lt))")
    elif args.center_only:
        reachability = CenterOnlyReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print("  Using CENTER-ONLY reachability (debugging mode)")
    elif args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print("  Using Lipschitz-based reachability (linear growth)")
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using sampling-based reachability ({args.samples} samples/dim)")
    
    rname = type(reachability).__name__
    output_dir = f"./results_adaptive/{rname}"
    
    adaptive = AdaptiveRefinement(
        args,
        env=env,
        gamma=args.gamma,
        cell_tree=cell_tree,
        reachability=reachability,
        output_dir=output_dir
    )
    
    start_time = time.time()
    adaptive.refine(
        epsilon=args.epsilon,
        max_refinements=args.refinements,
        vi_iterations_per_refinement=args.vi_iterations
    )
    elapsed = time.time() - start_time
    
    print(f"\n" + "="*70)
    print("ALGORITHM 2 COMPLETE")
    print("="*70)
    print(f"Total computation time: {elapsed:.2f} seconds")
    print(f"\nAll results saved to: {output_dir}/")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Safety Value Function Computation with Formal Guarantees (FIXED)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--algorithm', type=int, choices=[1, 2], required=True,
        help="Algorithm to run: 1 (basic discretization) or 2 (adaptive refinement)"
    )
    
    parser.add_argument('--velocity', type=float, default=1.0,
                       help="Constant velocity for Dubins car")
    parser.add_argument('--dt', type=float, default=0.1,
                       help="Time step for dynamics integration")
    parser.add_argument('--obstacle-radius', type=float, default=1.3,
                       help="Radius of circular obstacle")
    parser.add_argument('--gamma', type=float, default=0.1,
                       help="Discount factor (must satisfy γL_f < 1)")
    
    parser.add_argument('--resolution', type=int, default=10,
                       help="Grid resolution per dimension (Algorithm 1)")
    parser.add_argument('--iterations', type=int, default=200,
                       help="Maximum value iterations (Algorithm 1)")
    parser.add_argument('--tolerance', type=float, default=0.002,
                       help="Convergence tolerance (Algorithm 1)")
    parser.add_argument('--plot-freq', type=int, default=25,
                       help="Plot frequency in iterations (Algorithm 1)")
    
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="Error tolerance for refinement (Algorithm 2)")
    parser.add_argument('--initial-resolution', type=int, default=8,
                       help="Initial coarse grid resolution (Algorithm 2)")
    parser.add_argument('--refinements', type=int, default=25,
                       help="Maximum refinement iterations (Algorithm 2)")
    parser.add_argument('--vi-iterations', type=int, default=50,
                       help="VI iterations per refinement (Algorithm 2)")
    
    parser.add_argument('--samples', type=int, default=10,
                       help="Samples per dimension for reachability")
    parser.add_argument('--lipschitz-reach', action='store_true',
                       help="Use Lipschitz-based reachability (faster)")
    parser.add_argument('--center-only', action='store_true',
                   help="Use center-only reachability (debugging mode - INCORRECT)")
    parser.add_argument('--gronwall-reach', action='store_true',
                       help="Use Grönwall-based reachability (r*e^(Lt) expansion)")
    parser.add_argument('--use-inf-norm', action='store_true', default=True,
                       help="Use infinity norm for Grönwall radius (default: True)")
    parser.add_argument('--use-2-norm', dest='use_inf_norm', action='store_false',
                       help="Use 2-norm (Euclidean) for Grönwall radius")
    
    
    parser.add_argument('--plot-reachability', action='store_true',
                       help="Generate reachability visualization (Algorithm 1)")
    parser.add_argument('--plot-failure', action='store_true',
                       help="Generate failure function bounds plot (Algorithm 1)")
    parser.add_argument('--plot-FT-HJ', action='store_true')
    
    
    args = parser.parse_args()
    # CREATE GROUND TRUTH FIRST (before running algorithms)
    print("="*70)
    print("GENERATING GROUND TRUTH REFERENCE")
    print("="*70)
    if (args.plot_FT_HJ):
        env = DubinsCarEnvironment(
            v_const=args.velocity,
            dt=args.dt,
            obstacle_radius=args.obstacle_radius
        )
        
        # Compute ground truth with direct sampling
        ground_truth = compute_ground_truth_reachability(
            env,
            time_horizon=150,
            resolution=80 
        )
        
    # NOW run the selected algorithm
    if args.algorithm == 1:
        run_algorithm_1(args)
    else:
        run_algorithm_2(args)


if __name__ == "__main__":
    main()
    
    
    #python safety_value_function_3.py --algorithm 2 --resolution 10 --iterations 100 --lipschitz-reach --plot-failure  --plot-reachability
    
    
    
#local_value_iteration() only updates:

# Newly created cells from refinement
# Cells whose successor sets include any updated cell (backward reachability)



# python safety_value_function_3.py --algorithm 2 --resolution 1 --iterations 100 --lipschitz-reach --plot-failure  --plot-reachability --initial-resolution 51 

# python safety_value_function_3.py --algorithm 2 --resolution 1 --iterations 100 --lipschitz-reach --plot-failure  --plot-reachability --initial-resolution 1 --gamma 0.05 --dt 0.05
