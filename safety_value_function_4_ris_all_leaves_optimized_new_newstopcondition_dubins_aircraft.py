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


# ----------------------------------------------------------------------------
# Dubins Car Environment (Unchanged)
# ----------------------------------------------------------------------------
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
        
        # Lipschitz constants
        self.L_f = v_const
        self.L_l = np.sqrt(2)
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
        t_span = [0, self.dt]
        solution = odeint(self._dubins_ode, state, t_span, args=(action,), atol=1e-12, rtol=1e-12)
        state_next = solution[-1]
        # Normalize theta to [-π, π]
        theta_next = np.arctan2(np.sin(state_next[2]), np.cos(state_next[2]))
        state_next[2] = theta_next
        return state_next

    def failure_function(self, state: np.ndarray) -> float:
        pos = state[:2]
        dist_to_obstacle = np.linalg.norm(pos - self.obstacle_position)
        return (dist_to_obstacle - self.obstacle_radius)
    
    def get_lipschitz_constants(self) -> Tuple[float, float]:
        return self.L_f, self.L_l
    
    def get_state_dim(self) -> int:
        return 3

# ----------------------------------------------------------------------------
# ✈️ 3D Aircraft Evasion Environment (ODEINT-integrated)
# ----------------------------------------------------------------------------
class EvasionEnvironment(Environment):
    """
    3D evasion problem:
        ẋ₁ = -v + v*cos(x₃) + u*x₂
        ẋ₂ = v*sin(x₃) - u*x₁
        ẋ₃ = -u
    Collision if x₁² + x₂² ≤ 1.
    """
    def __init__(self, v_const=1.0, dt=0.05,
                 obstacle_position=(0.0, 0.0),
                 obstacle_radius=1.0, state_bounds=None):
        self.obstacle_position = np.array(obstacle_position)
        self.obstacle_radius = obstacle_radius
        self.v_const = v_const
        self.dt = dt

        if state_bounds is None:
            self.state_bounds = np.array([[-5.0, 5.0],
                                          [-5.0, 5.0],
                                          [-np.pi, np.pi]])
        else:
            self.state_bounds = state_bounds

        # Control discretization
        self.actions = np.linspace(-1.0, 1.0, 5)

        # Lipschitz constants
        self.L_f = 1 + self.v_const
        self.L_l = np.sqrt(2)

    def get_state_bounds(self) -> np.ndarray:
        return self.state_bounds

    def get_action_space(self) -> List:
        return self.actions.tolist()

    # ODE for odeint
    def _evasion_ode(self, state: np.ndarray, t: float, u: float) -> np.ndarray:
        x1, x2, x3 = state
        v = self.v_const
        dx1 = -v + v * np.cos(x3) + u * x2
        dx2 = v * np.sin(x3) - u * x1
        dx3 = -u
        return [dx1, dx2, dx3]

    def dynamics(self, state: np.ndarray, action: float) -> np.ndarray:
        t_span = [0, self.dt]
        sol = odeint(self._evasion_ode, state, t_span, args=(action,), atol=1e-12, rtol=1e-12)
        next_state = sol[-1]
        # Normalize θ to [-π, π] for consistency with Dubins
        next_state[2] = np.arctan2(np.sin(next_state[2]), np.cos(next_state[2]))
        return next_state

    def failure_function(self, state: np.ndarray) -> float:
        x1, x2, _ = state
        dx, dy = x1 - self.obstacle_position[0], x2 - self.obstacle_position[1]
        return np.sqrt(dx**2 + dy**2) - self.obstacle_radius

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
        
        # CHANGED: Remove V_lower_conservative, we'll modify V_lower directly
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
        
        import rtree
        
        # Create properties for 3D index
        p = rtree.index.Property()
        p.dimension = 3
        
        # Create new index with properties
        idx = rtree.index.Index(properties=p)
        self.cell_id_to_index = {}
        
        # Insert all leaf cells
        for i, cell in enumerate(self.leaves):
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
    
    def get_intersecting_cells(self, bounds: np.ndarray) -> List[Cell]:
        """
        Fast intersection query using spatial index.
        """
        if self.spatial_index is None:
            return [cell for cell in self.leaves if cell.intersects(bounds)]
        
        query_bbox = (
            bounds[0, 0], bounds[1, 0], bounds[2, 0],
            bounds[0, 1], bounds[1, 1], bounds[2, 1]
        )
        
        hit_indices = list(self.spatial_index.intersection(query_bbox))
        return [self.leaves[i] for i in hit_indices]
    
    def get_num_leaves(self) -> int:
        return len(self.leaves)


# ============================================================================
# PART 3: REACHABILITY ANALYZER
# ============================================================================

class GronwallReachabilityAnalyzer:
    """
    Reachability using Grönwall's inequality: expansion = r * e^(Lt)
    """
    
    def __init__(self, env, use_infinity_norm: bool = True, debug_verify: bool = False):
        self.env = env
        self.use_infinity_norm = use_infinity_norm
        self.debug_verify = debug_verify
        
        L_f, _ = env.get_lipschitz_constants()
        self.L = L_f
        self.dt = env.dt
        
        self.growth_factor = np.exp(self.L * self.dt)
        
        # Debug statistics
        self.debug_query_count = 0
        self.debug_mismatch_count = 0
        
        print(f"\nGrönwall Reachability Initialized:")
        print(f"  Lipschitz constant L = {self.L}")
        print(f"  Time step dt = {self.dt}")
        print(f"  Growth factor e^(Lt) = {self.growth_factor:.6f}")
    
    @staticmethod
    def fix_angle_interval(left_angle: float, right_angle: float) -> Tuple[float, float]:
        """Normalize an angle interval to a canonical form."""
        if abs(right_angle - left_angle) < 0.001:
            left_angle = right_angle - 0.01
        
        while right_angle < left_angle:
            right_angle += 2 * np.pi
        
        if right_angle - left_angle >= 2 * np.pi - 0.01:
            return -np.pi, np.pi
        
        while left_angle < -np.pi or right_angle < -np.pi:
            left_angle += 2 * np.pi
            right_angle += 2 * np.pi
        
        while left_angle > 3 * np.pi or right_angle > 3 * np.pi:
            left_angle -= 2 * np.pi
            right_angle -= 2 * np.pi
        
        while left_angle > np.pi and right_angle > np.pi:
            left_angle -= 2 * np.pi
            right_angle -= 2 * np.pi
        
        return left_angle, right_angle
    
    @staticmethod
    def intervals_intersect(a_left: float, a_right: float, 
                          b_left: float, b_right: float) -> bool:
        """Check if two angle intervals intersect."""
        return not (a_right < b_left or b_right < a_left)
    
    def compute_reachable_set(self, cell, action) -> np.ndarray:
        """Compute reachable set bounds using r * e^(Lt)."""
        center = cell.center
        
        if self.use_infinity_norm:
            r = 0.5 * cell.get_max_range()
        else:
            ranges = np.array([cell.get_range(j) for j in range(len(cell.bounds))])
            r = 0.5 * np.linalg.norm(ranges)
        
        center_next = self.env.dynamics(center, action)
        
        expansion = r * self.growth_factor
        
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        
        reach_bounds[0, :] = [center_next[0] - expansion, center_next[0] + expansion]
        reach_bounds[1, :] = [center_next[1] - expansion, center_next[1] + expansion]
        
        theta_lower = center_next[2] - expansion
        theta_upper = center_next[2] + expansion
        
        theta_lower, theta_upper = self.fix_angle_interval(theta_lower, theta_upper)
        reach_bounds[2, :] = [theta_lower, theta_upper]
        
        return reach_bounds
    
    def _check_theta_intersection(self, candidate: Cell, reach_bounds: np.ndarray) -> bool:
        """Helper: Check if theta dimension intersects."""
        cand_theta_left, cand_theta_right = self.fix_angle_interval(
            candidate.bounds[2, 0], 
            candidate.bounds[2, 1]
        )
        reach_theta_left = reach_bounds[2, 0]
        reach_theta_right = reach_bounds[2, 1]
        
        return self.intervals_intersect(
            cand_theta_left, cand_theta_right,
            reach_theta_left, reach_theta_right
        )
    
    def compute_successor_cells(self, cell, action, cell_tree) -> List[Cell]:
        """
        Find all cells that intersect the reachable set using spatial index.
        """
        reach_bounds = self.compute_reachable_set(cell, action)
        
        # Use spatial index for fast x,y intersection query
        candidates = cell_tree.get_intersecting_cells(reach_bounds)
        
        # Still need to manually check theta intersection
        successors_new = []
        for candidate in candidates:
            if self._check_theta_intersection(candidate, reach_bounds):
                successors_new.append(candidate)
        
        return successors_new
    
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
    V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, gamma = shared_data
    
    l_upper = l_upper_arr[cell_idx]
    l_lower = l_lower_arr[cell_idx]
    
    # Skip Bellman backup if this cell has no successors for any action
    if all(len(succ) == 0 for succ in succ_indices_by_action):
        return (cell_idx, V_upper_arr[cell_idx], V_lower_arr[cell_idx])
    
    best_min_val = -np.inf
    best_max_val = -np.inf
    
    for succ_indices in succ_indices_by_action:
        if len(succ_indices) == 0:
            continue
        
        # Direct numpy array indexing - VERY FAST
        upper_vals = V_upper_arr[succ_indices]
        lower_vals = V_lower_arr[succ_indices]
        
        action_lower = gamma * np.min(lower_vals)
        action_upper = gamma * np.max(upper_vals)
        
        if action_lower > best_min_val:
            best_min_val = action_lower
            best_max_val = action_upper
    
    new_V_lower = min(l_lower, best_min_val) if best_min_val > -np.inf else l_lower
    new_V_upper = min(l_upper, best_max_val) if best_max_val > -np.inf else l_upper
    
    return (cell_idx, new_V_upper, new_V_lower)


def _initialize_cell_worker(args):
    """
    Worker for parallel cell initialization.
    """
    cell_id, bounds, center, env, L_l = args
    
    # Compute failure function at center
    l_center = env.failure_function(center)
    
    # Compute radius
    ranges = bounds[:, 1] - bounds[:, 0]
    r = 0.5 * np.max(ranges)
    
    l_lower = l_center - L_l * r
    l_upper = l_center + L_l * r
    
    return (cell_id, l_lower, l_upper)


# ============================================================================
# PART 5: OPTIMIZED VALUE ITERATION (ALGORITHMS 1 & 3)
# ============================================================================

class SafetyValueIterator:
    """Implements Algorithm 1 & 3 with optimized parallelization and spatial indexing."""
    
    def __init__(self, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: GronwallReachabilityAnalyzer, output_dir: Optional[str] = None,
                 n_workers: Optional[int] = None, precompute_successors: bool = False):
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
        
        # Parallelization settings
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.precompute_successors_flag = precompute_successors
        self.successor_cache = {}
        self.pool = None
        
        print(f"Initialized with γ={gamma}, L_f={self.L_f}, L_l={self.L_l}")
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
        """Precompute all successor sets using spatial index."""
        print(f"\nPrecomputing successor sets with spatial index...")
        leaves = self.cell_tree.get_leaves()
        actions = self.env.get_action_space()
        
        print(f"  Computing successors for {len(leaves)} cells × {len(actions)} actions")
        start_time = time.time()
        
        for cell in leaves:
            for action in actions:
                successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
                key = (cell.cell_id, action)
                self.successor_cache[key] = [s.cell_id for s in successors]
        
        elapsed = time.time() - start_time
        print(f"  ✓ Precomputed {len(self.successor_cache)} successor sets in {elapsed:.2f}s")
    
    def initialize_cells(self):
        """Initialize stage cost bounds for all cells (parallelized)."""
        leaves = self.cell_tree.get_leaves()
        
        if len(leaves) == 0:
            return
        
        print(f"  Initializing {len(leaves)} cells in parallel...")
        start_time = time.time()
        
        # Prepare tasks
        tasks = [
            (cell.cell_id, cell.bounds.copy(), cell.center.copy(), self.env, self.L_l)
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
        for cell_id, l_lower, l_upper in results:
            cell = cell_dict[cell_id]
            cell.l_lower = l_lower
            cell.l_upper = l_upper
            cell.V_lower = cell.l_lower #check if make this inf instead 
            cell.V_upper = cell.l_upper #check if make this inf instead 
        
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
            (cell.cell_id, cell.bounds.copy(), cell.center.copy(), self.env, self.L_l)
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
        for cell_id, l_lower, l_upper in results:
            cell = cell_dict[cell_id]
            cell.l_lower = l_lower
            cell.l_upper = l_upper
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
            print(f"Conservative margin will be: ε_cons = (γ * δ_max) / (1 - γ) = ({self.gamma} * {delta_max}) / {1 - self.gamma}")
        else:
            print(f"Convergence tolerance: {convergence_tol}")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        
        for iteration in range(max_iterations):
            leaves = self.cell_tree.get_leaves()
            prev_upper = {cell.cell_id: cell.V_upper for cell in leaves}
            prev_lower = {cell.cell_id: cell.V_lower for cell in leaves}
            
            # Build compact numpy arrays for efficient parallel processing
            n_cells = len(leaves)
            cell_ids = np.array([c.cell_id for c in leaves])
            V_upper_arr = np.array([c.V_upper for c in leaves])
            V_lower_arr = np.array([c.V_lower for c in leaves])
            l_upper_arr = np.array([c.l_upper for c in leaves])
            l_lower_arr = np.array([c.l_lower for c in leaves])
            
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
            shared_data = (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, self.gamma)
            
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
                
                # Conservative stopping: δ^k ≥ -δ_max
                if delta_k >= -delta_max:
                    print(f"\n" + "="*60)
                    print("✓ CONSERVATIVE STOPPING CONDITION MET (Algorithm 3)")
                    print("="*60)
                    print(f"  δ^k = {delta_k:.10e} ≥ -δ_max = -{delta_max:.10e}")
                    
                    # Apply conservative correction to V_lower (Algorithm 3, line 29)
                    epsilon_cons = (self.gamma * delta_max) / (1 - self.gamma)
                    print(f"  Applying conservative margin: ε_cons = (γ * δ_max) / (1 - γ)")
                    print(f"  ε_cons = ({self.gamma} * {delta_max}) / {1 - self.gamma} = {epsilon_cons:.10e}")
                    print(f"  Correcting V_lower for {len(leaves)} cells: V_lower ← V_lower - ε_cons")
                    
                    for cell in leaves:
                        cell.V_lower = cell.V_lower - epsilon_cons
                    
                    print(f"  ✓ Conservative correction applied successfully")
                    self._save_plot(iteration + 1, final=True)
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
                    self._save_plot(iteration + 1, final=True)
                    break
            
            # Periodic plotting
            if plot_freq > 0 and (iteration + 1) % plot_freq == 0:
                self._save_plot(iteration + 1)
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
                epsilon_cons = (self.gamma * delta_max) / (1 - self.gamma)
                print(f"  Applying conservative margin: ε_cons = {epsilon_cons:.10e}")
                for cell in leaves:
                    cell.V_lower = cell.V_lower - epsilon_cons
                print(f"  ✓ Conservative correction applied to {len(leaves)} cells")
        
        print(f"\nValue iteration completed in {iteration + 1} iterations")
        return np.array(conv_history_upper), np.array(conv_history_lower)
    def local_value_iteration(self, updated_cells: Set[Cell], max_iterations: int = 100,
                        convergence_tol: float = 1e-3, conservative_mode: bool = False, 
                        delta_max: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        LOCAL value iteration: update ALL leaves after refinement.
        """
        leaves = self.cell_tree.get_leaves()
        
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
            
            shared_data = (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, self.gamma)
            
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
                    f"||V̄^k - V̄^k-1||_∞ = {diff_upper:10.20f}, "
                    f"||V_^k - V_^k-1||_∞ = {diff_lower:10.20f}, "
                    f"δ^k = {delta_k:12.20f}")
                
                if delta_k >= -delta_max:
                    print(f"    " + "="*40)
                    print(f"    ✓ CONSERVATIVE STOPPING (Local VI)")
                    print(f"    " + "="*40)
                    print(f"      δ^k = {delta_k:.20e} ≥ -δ_max = -{delta_max:.20e}")
                    
                    # Apply conservative correction
                    epsilon_cons = (self.gamma * delta_max) / (1 - self.gamma)
                    print(f"      Applying conservative margin: ε_cons = {epsilon_cons:.20e}")
                    print(f"      Correcting V_lower for {len(leaves)} cells")
                    
                    for cell in leaves:
                        cell.V_lower = cell.V_lower - epsilon_cons
                    print(f"      ✓ Conservative correction applied")
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
                epsilon_cons = (self.gamma * delta_max) / (1 - self.gamma)
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
        """Update successor cache after refinement."""
        if not new_cells:
            return
        
        print(f"  Updating successor cache for {len(new_cells)} new cells...")
        start_time = time.time()
        
        actions = self.env.get_action_space()
        
        # Identify refined parent cells
        refined_parent_ids = set()
        for cell in new_cells:
            if cell.parent is not None:
                refined_parent_ids.add(cell.parent.cell_id)
        
        # Remove all cache entries involving refined parents
        keys_to_delete = []
        for key in self.successor_cache.keys():
            cell_id, action = key
            successor_ids = self.successor_cache[key]
            
            if cell_id in refined_parent_ids or any(sid in refined_parent_ids for sid in successor_ids):
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.successor_cache[key]
        
        # Recompute successors for all affected cells
        affected_cells = set(new_cells)
        all_leaves = self.cell_tree.get_leaves()
        for cell in all_leaves:
            for action in actions:
                key = (cell.cell_id, action)
                if key not in self.successor_cache:
                    affected_cells.add(cell)
                    break
        
        for cell in affected_cells:
            for action in actions:
                successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
                key = (cell.cell_id, action)
                self.successor_cache[key] = [s.cell_id for s in successors]
        
        elapsed = time.time() - start_time
        print(f"  ✓ Cache updated in {elapsed:.2f}s")
            
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
                f"tol_{args.tolerance:.1e}_"
                f"eps_{args.epsilon:.3f}"
                f"vi-iterations_{args.vi_iterations}"
                f"conservative_{args.conservative}"
                f"delta-max_{args.delta_max}"
            )
            output_dir = os.path.join(
                "./results_adaptive_optimized_new_odeint_car_plane_new",
                f"{rname}_{param_suffix}"
            )
        self.output_dir = output_dir
        
        self.value_iterator = SafetyValueIterator(
            env, gamma, cell_tree, reachability, output_dir,
            n_workers=args.workers, precompute_successors=args.precompute
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
        conv_upper, conv_lower = self.value_iterator.value_iteration(
            max_iterations=vi_iterations_per_refinement,
            plot_freq=self.args.plot_freq,
            conservative_mode=self.args.conservative,
            delta_max=self.args.delta_max
        )
        
        # Print initial convergence summary
        if len(conv_upper) > 0:
            print(f"Initial VI completed in {len(conv_upper)} iterations")
            print(f"Final convergence: ||V̄||_∞ = {conv_upper[-1]:.8e}, ||V_||_∞ = {conv_lower[-1]:.8e}")
        
        # Save plot after initial VI
        filename = os.path.join(self.output_dir, "value_function_phase_0_complete.png")
        plot_value_function(self.env, self.cell_tree, filename, 0)
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
            index_start = time.time()
            self.cell_tree.rebuild_spatial_index()
            index_time = time.time() - index_start
            print(f"    Spatial index rebuilt in {index_time:.3f}s")
            
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
            
            # Local VI convergence summary
            print(f"  Local VI completed in {len(conv_upper)} iterations, time: {local_vi_time:.3f}s")
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
            phase_time = time.time() - refinement_start
            print(f"  Phase {refinement_iter + 1} completed in {phase_time:.3f}s")
            
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
        theta_slices = [0, np.pi/4, np.pi/2, np.pi]
    
    n_slices = len(theta_slices)
    fig, axes = plt.subplots(3, n_slices, figsize=(5*n_slices, 14))
    
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
    """Plot value function by directly coloring leaf cells."""
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
            values.append(cell. V_upper)
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
    
    # NEW: Print value statistics to terminal
    # if upper:
    #     print(f"    V̄_γ statistics (θ={theta:.2f}): min={min(values):.40f}, max={max(values):.40f}")
    # else:
    #     print(f"    V_γ statistics (θ={theta:.2f}): min={min(values):.40f}, max={max(values):.40f}")
    
    # Determine color scale
    # Determine color scale
    # Determine color scale
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    cmap = plt.cm.RdYlGn

    # --- FIXED COLOR NORMALIZATION ---
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    cmap = plt.cm.RdYlGn

    vmin, vmax = np.min(values), np.max(values)

    # If all values are the same, expand slightly
    if np.isclose(vmin, vmax):
        vmin, vmax = vmin - 1e-12, vmax + 1e-12

    # If all values are negative, still use full colormap range (no white)
    if vmax <= 0:
        # shift range so that colormap still covers dark to bright
        norm = Normalize(vmin=vmin, vmax=0)
    elif vmin >= 0:
        norm = Normalize(vmin=0, vmax=vmax)
    else:
        # mixed sign values
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Plot each cell as a colored rectangle
    for cell, value in relevant_cells:
        a_x, b_x = cell.bounds[0]
        a_y, b_y = cell.bounds[1]
        color = cmap(norm(value))
        rect = Rectangle((a_x, a_y), b_x - a_x, b_y - a_y,
                         facecolor=color, edgecolor='black', linewidth=1, alpha=1.0)
        ax.add_patch(rect)

    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          facecolor='none', edgecolor='darkblue', linewidth=2, zorder=10)
    elif isinstance(env, EvasionEnvironment):
        obstacle = Circle((0, 0), env.obstacle_radius,
                          facecolor='none', edgecolor='darkred', linestyle='--', linewidth=2, zorder=10)
    ax.add_patch(obstacle)

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

    # --- Use scientific notation ---
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1e}'))
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    # --- Set evenly spaced ticks, excluding the very top (vmax) ---
    tick_vals = np.linspace(vmin, vmax, 5)
    tick_vals = tick_vals[:-1]  # remove top tick to avoid overlap
    # cbar.set_ticks(tick_vals)
    # cbar.update_ticks()

    # optional: pad labels slightly away from bar to avoid clipping
    # cbar.ax.tick_params(pad=2)


    # Explicit ticks (optional)
    cbar.ax.yaxis.offsetText.set_visible(False)
    def safe_ticks(vmin, vmax, zero_threshold=0.1):
        """
        Plot vmin, vmax, and optionally 0 if both are sufficiently far from 0.
        """
        ticks = [vmin, vmax]

        # include 0 only if both sides are at least ±zero_threshold away from zero
        if abs(vmin) >= zero_threshold and abs(vmax) >= zero_threshold:
            ticks.insert(1, 0.0)

        # Ensure unique and sorted ticks (helps with colorbar orientation)
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
        elif cell.V_upper < 0:
            color = C_UNSAFE
        else:
            color = C_BOUND
        
        # Draw colored rectangle
        rect = Rectangle(
            (cell.bounds[0, 0], cell.bounds[1, 0]),
            cell.get_range(0), cell.get_range(1),
            facecolor=color, edgecolor='none', alpha=1.0
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            facecolor='none',
            edgecolor='darkblue',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(obstacle)
    elif isinstance(env, EvasionEnvironment):
        obstacle = Circle(
            (0, 0),
            env.obstacle_radius,
            facecolor='none',
            edgecolor='darkred',
            linestyle='--',
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
        Patch(facecolor=C_SAFE, edgecolor='black', label='Safe (V_>0)'),
        Patch(facecolor=C_UNSAFE, edgecolor='black', label='Unsafe (V̄<0)'),
        Patch(facecolor=C_BOUND, edgecolor='black', label='Boundary (mixed)')
    ]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)


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
    value_iter = SafetyValueIterator(env=env, gamma=args.gamma, cell_tree=cell_tree, reachability=reachability,output_dir=f"./results/algorithm1_dynamics_{args.dynamics}_resol_{args.resolution},tol_{args.tolerance:.1e}")
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
    L_f, _ = env.get_lipschitz_constants()
    if args.gamma * L_f >= 1:
        raise ValueError(f"Contraction condition violated: γL_f = {args.gamma * L_f} >= 1")
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
                       help="Time step for dynamics integration")
    # parser.add_argument('--obstacle-radius', type=float, default=1.3,
    #                    help="Radius of circular obstacle (Dubins only)")
    parser.add_argument('--gamma', type=float, default=0.1,
                       help="Discount factor (must satisfy γL_f < 1)")
    
    # Algorithm parameters
    parser.add_argument('--resolution', type=int, default=10,
                       help="Grid resolution per dimension (Algorithm 1)")
    parser.add_argument('--iterations', type=int, default=200,
                       help="Maximum value iterations (Algorithm 1)")
    parser.add_argument('--tolerance', type=float, default=1e-15,
                       help="Convergence tolerance")
    parser.add_argument('--plot-freq', type=int, default=1000, #dont create intermediate plots
                       help="Plot frequency in iterations (Algorithm 1)")
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="Error tolerance for refinement (Algorithm 2)")
    parser.add_argument('--initial-resolution', type=int, default=8,
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
            v_const=args.velocity, dt=args.dt, obstacle_radius=1.3
        )
    else:
        env = EvasionEnvironment(v_const=args.velocity, dt=args.dt,obstacle_position=(0.0, 0.0),
                obstacle_radius=1.0)

    # Continue as before
    if args.algorithm == 1:
        run_algorithm_1(args, env)
    else:
        run_algorithm_2(args, env)


if __name__ == "__main__":
    main()
    
    
# python safety_value_function_4_ris_all_leaves_optimized_new_newstopcondition_dubins_aircraft.py \
#   --algorithm 2 \
#   --dynamics dubins \
#   --gamma 0.2 \
#   --dt 0.05 \
#   --velocity 1.0 \
#   --epsilon 0.04 \
#   --vi-iterations 1000 \
#  --initial-resolution 1

# python safety_value_function_4_ris_all_leaves_optimized_new_newstopcondition_dubins_aircraft.py \
#   --algorithm 2 \
#   --dynamics evasion \
#   --gamma 0.49 \
#   --dt 0.05 \
#   --velocity 1.0 \
#   --epsilon 0.04 \
#   --vi-iterations 1000 \
#  --initial-resolution 1






    
    '''
    
    def test_lipschitz_constants(env, n_samples=10000):
        """Empirically verify Lipschitz constants"""
        
        max_L_f_observed = 0
        max_L_l_observed = 0
        
        for _ in range(n_samples):
            # Random states
            x = np.random.uniform(env.state_bounds[:, 0], env.state_bounds[:, 1])
            x_prime = np.random.uniform(env.state_bounds[:, 0], env.state_bounds[:, 1])
            
            # Random action
            u = np.random.choice(env.actions)
            
            # Test L_f
            f_x = env._evasion_ode(x, 0, u)
            f_x_prime = env._evasion_ode(x_prime, 0, u)
            
            state_diff_inf = np.max(np.abs(x - x_prime))
            dynamics_diff_inf = np.max(np.abs(np.array(f_x) - np.array(f_x_prime)))
            
            if state_diff_inf > 1e-6:  # Avoid division by near-zero
                L_f_sample = dynamics_diff_inf / state_diff_inf
                max_L_f_observed = max(max_L_f_observed, L_f_sample)
            
            # Test L_l
            l_x = env.failure_function(x)
            l_x_prime = env.failure_function(x_prime)
            
            failure_diff = abs(l_x - l_x_prime)
            
            if state_diff_inf > 1e-6:
                L_l_sample = failure_diff / state_diff_inf
                max_L_l_observed = max(max_L_l_observed, L_l_sample)
        
        print(f"Observed max L_f: {max_L_f_observed:.6f} (claimed: {env.L_f})")
        print(f"Observed max L_l: {max_L_l_observed:.6f} (claimed: {env.L_l})")
        
        return max_L_f_observed, max_L_l_observed

    # Test it
    env = EvasionEnvironment(v_const=1.0)
    test_lipschitz_constants(env)
    #Observed max L_f: 1.903273 (claimed: 2.0)
    #Observed max L_l: 1.397104 (claimed: 1.4142135623730951)
    

    def test_lipschitz_constants(env, n_samples=10000):
        """Empirically verify Lipschitz constants"""
        
        max_L_f_observed = 0
        max_L_l_observed = 0
        
        for _ in range(n_samples):
            # Random states
            x = np.random.uniform(env.state_bounds[:, 0], env.state_bounds[:, 1])
            x_prime = np.random.uniform(env.state_bounds[:, 0], env.state_bounds[:, 1])
            
            # Random action
            u = np.random.choice(env.actions)
            
            # Test L_f
            f_x = env._dubins_ode(x, 0, u)
            f_x_prime = env._dubins_ode(x_prime, 0, u)
            
            state_diff_inf = np.max(np.abs(x - x_prime))
            dynamics_diff_inf = np.max(np.abs(np.array(f_x) - np.array(f_x_prime)))
            
            if state_diff_inf > 1e-6:  # Avoid division by near-zero
                L_f_sample = dynamics_diff_inf / state_diff_inf
                max_L_f_observed = max(max_L_f_observed, L_f_sample)
            
            # Test L_l
            l_x = env.failure_function(x)
            l_x_prime = env.failure_function(x_prime)
            
            failure_diff = abs(l_x - l_x_prime)
            
            if state_diff_inf > 1e-6:
                L_l_sample = failure_diff / state_diff_inf
                max_L_l_observed = max(max_L_l_observed, L_l_sample)
        
        print(f"Observed max L_f: {max_L_f_observed:.6f} (claimed: {env.L_f})")
        print(f"Observed max L_l: {max_L_l_observed:.6f} (claimed: {env.L_l})")
        
        return max_L_f_observed, max_L_l_observed


    # Test it
    env = DubinsCarEnvironment(v_const=1.0)
    # Observed max L_f: 0.990269 (claimed: 1.0)
    # Observed max L_l: 1.410656 (claimed: 1)
    test_lipschitz_constants(env)
    '''
    
    
#     python safety_value_function_4_ris_all_leaves_optimized_new_newstopcondition_dubins_aircraft.py \
#   --algorithm 2 \
#   --dynamics evasion \
#   --gamma 0.49 \
#   --dt 0.05 \
#   --velocity 1.0 \
#   --epsilon 0.04 \
#   --vi-iterations 1000 \
#  --initial-resolution 1 \
#   --conservative \
#   --delta-max 1e-13