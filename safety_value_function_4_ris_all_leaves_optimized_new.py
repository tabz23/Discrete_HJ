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
    python safety_value_function_5.py --algorithm 2 --resolution 1 --iterations 100 --plot-failure --initial-resolution 1 --gamma 0.05 --dt 0.05
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
        return (dist_to_obstacle - self.obstacle_radius)
    
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
    """Tree structure to manage cells with adaptive refinement and spatial indexing."""
    
    def __init__(self, initial_bounds: np.ndarray, initial_resolution: int = 10):
        self.root_bounds = initial_bounds
        self.dim = len(initial_bounds)
        self.next_id = 0
        self.leaves = []
        self.all_cells = []
        
        # Spatial index for fast intersection queries
        self.spatial_index = None
        self.cell_id_to_index = {}  # Maps cell_id -> rtree index
        
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
        
        # Create new index with explicit dimension specification
        import rtree
        
        # Create properties for 3D index
        p = rtree.index.Property()
        p.dimension = 3  # Explicitly set to 3D for (x, y, theta)
        
        # Create new index with properties
        idx = rtree.index.Index(properties=p)
        self.cell_id_to_index = {}
        
        # Insert all leaf cells
        for i, cell in enumerate(self.leaves):
            # Create bounding box as a flat list/tuple
            # Format: (xmin, ymin, theta_min, xmax, ymax, theta_max)
            bbox = (
                cell.bounds[0, 0],  # xmin
                cell.bounds[1, 0],  # ymin
                cell.bounds[2, 0],  # theta_min
                cell.bounds[0, 1],  # xmax
                cell.bounds[1, 1],  # ymax
                cell.bounds[2, 1]   # theta_max
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
        
        Args:
            bounds: numpy array of shape (dim, 2) with [min, max] for each dimension
        
        Returns:
            List of cells that intersect the given bounds
        """
        if self.spatial_index is None:
            # Fallback to linear search if index not built
            return [cell for cell in self.leaves if cell.intersects(bounds)]
        
        # Query bounding box as a flat tuple
        # Format: (xmin, ymin, theta_min, xmax, ymax, theta_max)
        query_bbox = (
            bounds[0, 0],  # xmin
            bounds[1, 0],  # ymin
            bounds[2, 0],  # theta_min
            bounds[0, 1],  # xmax
            bounds[1, 1],  # ymax
            bounds[2, 1]   # theta_max
        )
        
        # Get indices of intersecting cells
        hit_indices = list(self.spatial_index.intersection(query_bbox))
        
        # Return corresponding cell objects
        return [self.leaves[i] for i in hit_indices]
    def get_cell_containing_point(self, point: np.ndarray) -> Optional[Cell]:
        """Find the LEAF cell containing the point by traversing the tree."""
        for cell in self.leaves:
            if cell.contains_point(point):
                return cell
        return None
    
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
        self.debug_verify = debug_verify  # Enable correctness verification
        
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
        print(f"  Norm type: {'∞-norm (max)' if use_infinity_norm else '2-norm (Euclidean)'}")
        print(f"  Debug verification: {'ENABLED' if debug_verify else 'DISABLED'}")
    
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
    
    def compute_successor_cells_old(self, cell, action, cell_tree) -> List[Cell]:
        """OLD METHOD: Linear search through all cells (for verification only)."""
        reach_bounds = self.compute_reachable_set(cell, action)
        
        successors = []
        
        for candidate in cell_tree.get_leaves():
            if (candidate.bounds[0, 1] < reach_bounds[0, 0] or 
                candidate.bounds[0, 0] > reach_bounds[0, 1]):
                continue
            
            if (candidate.bounds[1, 1] < reach_bounds[1, 0] or 
                candidate.bounds[1, 0] > reach_bounds[1, 1]):
                continue
            
            if not self._check_theta_intersection(candidate, reach_bounds):
                continue
            
            successors.append(candidate)
        
        return successors
    
    def compute_successor_cells(self, cell, action, cell_tree) -> List[Cell]:
        """
        NEW METHOD: Find all cells that intersect the reachable set using spatial index.
        
        Includes debug verification against old linear search method.
        """
        reach_bounds = self.compute_reachable_set(cell, action)
        
        # NEW METHOD: Use spatial index for fast x,y intersection query
        candidates = cell_tree.get_intersecting_cells(reach_bounds)
        
        # Still need to manually check theta intersection
        successors_new = []
        reach_theta_left, reach_theta_right = reach_bounds[2, 0], reach_bounds[2, 1]
        
        for candidate in candidates:
            if self._check_theta_intersection(candidate, reach_bounds):
                successors_new.append(candidate)
        
        # DEBUG VERIFICATION: Compare with old method
        if self.debug_verify:
            self.debug_query_count += 1
            
            # Compute using old linear search method
            successors_old = self.compute_successor_cells_old(cell, action, cell_tree)
            
            # Compare results
            new_ids = set(s.cell_id for s in successors_new)
            old_ids = set(s.cell_id for s in successors_old)
            
            if new_ids != old_ids:
                self.debug_mismatch_count += 1
                print(f"\n⚠️  DEBUG MISMATCH #{self.debug_mismatch_count}:")
                print(f"  Cell: {cell.cell_id}, Action: {action}")
                print(f"  Old method found: {len(successors_old)} successors")
                print(f"  New method found: {len(successors_new)} successors")
                print(f"  Missing in new: {old_ids - new_ids}")
                print(f"  Extra in new: {new_ids - old_ids}")
                print(f"  Reach bounds: {reach_bounds}")
                raise AssertionError("Spatial index returned different results!")
            
            # Periodic progress report
            if self.debug_query_count % 1000 == 0:
                print(f"  [DEBUG] Verified {self.debug_query_count} successor queries - All correct ✓")
        
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
            else:
                print(f"⚠️  {self.debug_mismatch_count} mismatches detected!")
            print(f"{'='*70}\n")


# ============================================================================
# PART 4: OPTIMIZED PARALLEL WORKERS
# ============================================================================

def _bellman_update_optimized(task, shared_data):
    """
    Optimized worker: uses numpy arrays instead of serialized objects.
    
    Args:
        task: (cell_idx, succ_indices_by_action)
        shared_data: (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, gamma)
    """
    cell_idx, succ_indices_by_action = task
    V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, gamma = shared_data
    
    l_upper = l_upper_arr[cell_idx]
    l_lower = l_lower_arr[cell_idx]
    
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
    
    Args:
        args: (cell_id, bounds, center, env, L_l)
    
    Returns:
        (cell_id, l_lower, l_upper)
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


def _precompute_successor_worker(args):
    """Worker for parallel successor set precomputation."""
    cell_data, action, env, reachability, all_leaves_data = args
    
    cell_id, bounds, center, _, _, _, _ = cell_data
    
    # Reconstruct cell
    cell = Cell(bounds, cell_id)
    cell.center = center
    
    # Reconstruct all leaves
    all_leaves = []
    for leaf_data in all_leaves_data:
        leaf_id, leaf_bounds, leaf_center, _, _, _, _ = leaf_data
        leaf_cell = Cell(leaf_bounds, leaf_id)
        leaf_cell.center = leaf_center
        all_leaves.append(leaf_cell)
    
    # Create temporary cell tree
    class TempCellTree:
        def get_leaves(self):
            return all_leaves
    
    temp_tree = TempCellTree()
    
    # Compute successors
    successors = reachability.compute_successor_cells(cell, action, temp_tree)
    successor_ids = [s.cell_id for s in successors]
    
    return ((cell_id, action), successor_ids)


# ============================================================================
# PART 5: OPTIMIZED VALUE ITERATION
# ============================================================================

class SafetyValueIterator:
    """Implements Algorithm 1 with optimized parallelization and spatial indexing."""
    
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
        self.pool = None  # Persistent worker pool
        
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
    
    def _serialize_cell(self, cell: Cell):
        """Serialize cell for multiprocessing."""
        return (
            cell.cell_id,
            cell.bounds.copy(),
            cell.center.copy(),
            cell.V_upper,
            cell.V_lower,
            cell.l_upper,
            cell.l_lower
        )
    
    def _precompute_all_successors(self):
        """Precompute all successor sets using spatial index."""
        print(f"\nPrecomputing successor sets with spatial index...")
        leaves = self.cell_tree.get_leaves()
        actions = self.env.get_action_space()
        
        print(f"  Computing successors for {len(leaves)} cells × {len(actions)} actions = {len(leaves) * len(actions)} queries")
        start_time = time.time()
        
        # Use spatial index directly - no need for parallel precomputation
        for cell in leaves:
            for action in actions:
                successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
                key = (cell.cell_id, action)
                self.successor_cache[key] = [s.cell_id for s in successors]
        
        elapsed = time.time() - start_time
        print(f"  ✓ Precomputed {len(self.successor_cache)} successor sets in {elapsed:.2f}s")
        print(f"  Average: {len(self.successor_cache)/elapsed:.1f} queries/second")
    
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
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
        
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
            cell.V_lower = -np.inf
            cell.V_upper = np.inf
        
        elapsed = time.time() - start_time
        print(f"  ✓ Initialized in {elapsed:.2f}s ({len(new_cells)/elapsed:.1f} cells/s)")
    
    def value_iteration(self, max_iterations: int = 1000, convergence_tol: float = 1e-3,
                       plot_freq: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Run value iteration until convergence (optimized parallelization)."""
        self.initialize_cells()
        if not self.successor_cache:
            print("  Precomputing successor sets (required for Algorithm 1)...")
            self._precompute_all_successors()
        
        # Precompute successors if not already done
        if self.precompute_successors_flag and not self.successor_cache:
            self._precompute_all_successors()
        
        # CREATE PERSISTENT POOL ONCE
        if self.pool is None:
            self.pool = Pool(self.n_workers)
        
        conv_history_upper = []
        conv_history_lower = []
        
        print(f"\nStarting OPTIMIZED PARALLEL value iteration (max {max_iterations} iterations)...")
        print(f"Convergence tolerance: {convergence_tol}")
        print(f"Number of cells: {self.cell_tree.get_num_leaves()}")
        print(f"Workers: {self.n_workers}")
        
        for iteration in range(max_iterations):
            leaves = self.cell_tree.get_leaves()
            prev_upper = {cell.cell_id: cell.V_upper for cell in leaves}
            prev_lower = {cell.cell_id: cell.V_lower for cell in leaves}
            
            # **# **OPTIMIZED**: Only send what workers need
            # Build compact numpy arrays instead of serializing objects
            n_cells = len(leaves)
            cell_ids = np.array([c.cell_id for c in leaves])
            V_upper_arr = np.array([c.V_upper for c in leaves])
            V_lower_arr = np.array([c.V_lower for c in leaves])
            l_upper_arr = np.array([c.l_upper for c in leaves])
            l_lower_arr = np.array([c.l_lower for c in leaves])
            
            # Build id->index mapping
            id_to_idx = {cid: idx for idx, cid in enumerate(cell_ids)}
            
            # Prepare compact tasks with indices instead of full cell data
            tasks = []
            for idx, cell in enumerate(leaves):
                # Get successor indices for this cell
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
            
            # Shared data (passed once per iteration, not per task)
            shared_data = (V_upper_arr, V_lower_arr, l_upper_arr, l_lower_arr, self.gamma)
            
            # Parallel Bellman updates with MINIMAL data transfer
            worker_func = partial(_bellman_update_optimized, shared_data=shared_data)
            results = self.pool.map(worker_func, tasks)
            
            # Apply updates
            for cell_idx, new_upper, new_lower in results:
                leaves[cell_idx].V_upper = new_upper
                leaves[cell_idx].V_lower = new_lower
            
            # Check convergence
            diff_upper = max(abs(leaves[i].V_upper - prev_upper[leaves[i].cell_id]) for i in range(n_cells))
            diff_lower = max(abs(leaves[i].V_lower - prev_lower[leaves[i].cell_id]) for i in range(n_cells))
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"Iteration {iteration + 1}: "
                  f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.20f}, "
                  f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.20f}")
            
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
        LOCAL value iteration: update ALL leaves after refinement.
        
        SIMPLIFIED: Updates all cells since refinement affects the global value function.
        """
        leaves = self.cell_tree.get_leaves()
        
        print(f"  Local VI: updating all {len(leaves)} cells")
        print(f"    (including {len(updated_cells)} newly created cells)")
        
        if len(leaves) == 0:
            print("    No cells to update!")
            return np.array([]), np.array([])
        
        # Update successor cache for new cells
        if updated_cells:
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
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"    Iter {iteration + 1}: "
                f"||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.20f}, "
                f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.20f}")
            
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"    ✓ Local VI converged at iteration {iteration + 1}!")
                break
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
        
    def _update_successor_cache_for_new_cells(self, new_cells: Set[Cell]):
        """Update successor cache with new cells after refinement using spatial index."""
        if not new_cells:
            return
        
        print(f"  Updating successor cache for {len(new_cells)} new cells (with spatial index)...")
        start_time = time.time()
        
        actions = self.env.get_action_space()
        
        # NO LONGER NEED TO SERIALIZE ALL LEAVES - spatial index handles it!
        # Just compute successors directly using spatial index
        new_entries = {}
        for cell in new_cells:
            for action in actions:
                successors = self.reachability.compute_successor_cells(
                    cell, action, self.cell_tree
                )
                key = (cell.cell_id, action)
                new_entries[key] = [s.cell_id for s in successors]
        
        # Update cache
        self.successor_cache.update(new_entries)
        
        elapsed = time.time() - start_time
        print(f"  ✓ Updated cache in {elapsed:.2f}s (total cached: {len(self.successor_cache)})")
        print(f"    Average: {len(new_entries)/elapsed:.1f} queries/second")
        
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
# PART 6: ADAPTIVE REFINEMENT (ALGORITHM 2/3)
# ============================================================================

class AdaptiveRefinement:
    """Algorithm 2/3 with optimized parallelization and spatial indexing."""
    
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
                f"gamma_{gamma:.3f}_"
                f"dt_{env.dt:.3f}_"
                f"tol_{args.tolerance:.1e}_"
                f"eps_{args.epsilon:.3f}"
                f"vi-iterations_{args.vi_iterations}"
            )
            output_dir = os.path.join(
                "./results_adaptive_optimized_new",
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
        """Main adaptive refinement loop with parallelization and spatial indexing."""
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
            
            # CRITICAL: Rebuild spatial index after refinements
            print(f"\nRebuilding spatial index after refinement...")
            self.cell_tree.rebuild_spatial_index()
            
            # Initialize new cells
            if new_cells:
                print(f"\nInitializing {len(new_cells)} new cells...")
                self.value_iterator.initialize_new_cells(new_cells)
            
            # Set refinement phase BEFORE LOCAL VI
            self.value_iterator.refinement_phase = refinement_iter + 1
            
            # LOCAL value iteration
            print(f"\nLocal Value Iteration (Phase {refinement_iter + 1}):")
            
            conv_upper, conv_lower = self.value_iterator.local_value_iteration(
                updated_cells=set(new_cells),
                max_iterations=vi_iterations_per_refinement,
                convergence_tol=self.args.tolerance
            )
            
            # Save plot AFTER local VI completes
            filename = os.path.join(
                self.output_dir, 
                f"value_function_phase_{refinement_iter + 1}_complete.png"
            )
            plot_value_function(self.env, self.cell_tree, filename, refinement_iter + 1)
            
            # Post-VI queue state
            new_boundary = self._identify_boundary_cells()
            new_refinable = [c for c in new_boundary if c.get_max_range() > eta_min]
            
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
        
        # Print debug verification summary
        self.reachability.print_debug_summary()
        
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
    epsilon = 1e-6
    
    if abs(vmax - vmin) < 1e-10:
        if abs(vmin) < 1e-10:
            norm = Normalize(vmin=-0.1, vmax=0.1)
        else:
            norm = Normalize(vmin=vmin-0.1*abs(vmin), vmax=vmax+0.1*abs(vmax))
    elif vmin > 0:
        norm = TwoSlopeNorm(vmin=0, vcenter=max(0, vmin-epsilon), vmax=vmax)
    elif vmax < 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=min(0, vmax+epsilon), vmax=0)
    elif vmin == 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=0, vcenter=epsilon, vmax=vmax)
    elif vmin < 0 and vmax == 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=-epsilon, vmax=0)
    elif vmin == 0 and vmax == 0:
        norm = Normalize(vmin=-0.1, vmax=0.1)
    else:
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
    edgecolor='none',  # No edge
    linewidth=0,       # No width
    alpha=1.0
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            facecolor='none',        # CHANGED: No fill color
            edgecolor='darkblue',    # Just the edge
            linewidth=2,             # Thickness of the edge
            zorder=10
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
    
    cbar = plt.colorbar(sm, ax=ax, label=label)
    
    # Set explicit tick locations
    if vmin < 0 and vmax > 0:
        tick_vals = [vmin, vmin/2, 0, vmax/2, vmax]
    elif vmin >= 0:
        tick_vals = [0, vmax/3, 2*vmax/3, vmax]
    elif vmax <= 0:
        tick_vals = [vmin, 2*vmin/3, vmin/3, 0]
    else:
        tick_vals = [vmin, 0, vmax]
    
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v:.20f}' for v in tick_vals])


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
      edgecolor='none',  # No edge
    linewidth=0,       # No width
    alpha=1.0
        )
        ax.add_patch(rect)
    
    # Draw obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(
            env.obstacle_position,
            env.obstacle_radius,
            facecolor='none',        # CHANGED: No fill color
            edgecolor='darkblue',    # Just the edge
            linewidth=2,             # Thickness of the edge
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
    edgecolor='none',  # No edge
    linewidth=0,       # No width
    alpha=1.0
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
                facecolor='none',        # CHANGED: No fill color
                edgecolor='darkblue',    # Just the edge
                linewidth=2,             # Thickness of the edge
                zorder=10
            )
            ax.add_patch(obstacle)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(title)
        
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(r"$\ell$ value", rotation=270, labelpad=15)
    
    plt.tight_layout()
    filename = os.path.join(save_dir, f"{filename_prefix}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved ℓ-bounds plot to {filename}")


# ============================================================================
# PART 8: MAIN INTERFACE
# ============================================================================

def run_algorithm_1(args):
    """Runs Algorithm 1: Basic discretization with value iteration (optimized)."""
    print("="*70)
    print("ALGORITHM 1: Discretization Routine (OPTIMIZED WITH SPATIAL INDEX)")
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
    
    # Reachability analyzer with debug verification
    reachability = GronwallReachabilityAnalyzer(
        env, 
        use_infinity_norm=args.use_inf_norm,
        debug_verify=args.debug_verify
    )
    
    value_iter = SafetyValueIterator(
        env=env,
        gamma=args.gamma,
        cell_tree=cell_tree,
        reachability=reachability,
        output_dir=None,
        n_workers=args.workers,
        precompute_successors=args.precompute
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
    
    # Print debug summary
    reachability.print_debug_summary()
    
    plot_convergence(conv_upper, conv_lower, f"{value_iter.output_dir}/convergence.png")
    print(f"\nAll results saved to: {value_iter.output_dir}/")


def run_algorithm_2(args):
    """Runs Algorithm 2/3: Adaptive refinement (optimized with spatial index)."""
    print("="*70)
    print("ALGORITHM 2/3: Adaptive Refinement (OPTIMIZED WITH SPATIAL INDEX)")
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
    
    # Reachability analyzer with debug verification
    reachability = GronwallReachabilityAnalyzer(
        env, 
        use_infinity_norm=args.use_inf_norm,
        debug_verify=args.debug_verify
    )
    
    rname = type(reachability).__name__
    param_suffix = (
        f"gamma_{args.gamma:.3f}_"
        f"dt_{env.dt:.3f}_"
        f"tol_{args.tolerance:.1e}_"
        f"eps_{args.epsilon:.3f}"
        f"vi-iterations_{args.vi_iterations}"
    )
    output_dir = os.path.join(
        "./results_adaptive_optimized_new",
        f"{rname}_{param_suffix}"
    )
    
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
        description="Safety Value Function Computation - OPTIMIZED WITH SPATIAL INDEX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    parser.add_argument('--gamma', type=float, default=0.1,
                       help="Discount factor (must satisfy γL_f < 1)")
    
    # Algorithm 1 parameters
    parser.add_argument('--resolution', type=int, default=10,
                       help="Grid resolution per dimension (Algorithm 1)")
    parser.add_argument('--iterations', type=int, default=200,
                       help="Maximum value iterations (Algorithm 1)")
    parser.add_argument('--tolerance', type=float, default=0.002,
                       help="Convergence tolerance")
    parser.add_argument('--plot-freq', type=int, default=25,
                       help="Plot frequency in iterations (Algorithm 1)")
    
    # Algorithm 2 parameters
    parser.add_argument('--epsilon', type=float, default=0.1,
                       help="Error tolerance for refinement (Algorithm 2)")
    parser.add_argument('--initial-resolution', type=int, default=8,
                       help="Initial coarse grid resolution (Algorithm 2)")
    parser.add_argument('--refinements', type=int, default=27,
                       help="Maximum refinement iterations (Algorithm 2)")
    parser.add_argument('--vi-iterations', type=int, default=50,
                       help="VI iterations per refinement (Algorithm 2)")
    
    # Reachability parameters
    parser.add_argument('--use-inf-norm', action='store_true', default=True,
                       help="Use infinity norm for Grönwall radius (default: True)")
    parser.add_argument('--use-2-norm', dest='use_inf_norm', action='store_false',
                       help="Use 2-norm (Euclidean) for Grönwall radius")
    
    # Parallelization parameters
    parser.add_argument('--workers', type=int, default=None,
                       help="Number of parallel workers (default: CPU count - 1)")
    parser.add_argument('--precompute', action='store_true',
                       help="Precompute all successor sets before VI (recommended for large grids)")
    
    # Visualization parameters
    parser.add_argument('--plot-failure', action='store_true',
                       help="Generate failure function bounds plot")
    
    # Debug parameters
    parser.add_argument('--debug-verify', action='store_true',
                       help="Enable debug verification: compare spatial index with linear search (slower)")
    
    args = parser.parse_args()
    
    # Set default workers if not specified
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    print(f"\n{'='*70}")
    print("OPTIMIZED SAFETY VALUE FUNCTION WITH SPATIAL INDEX")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Algorithm: {args.algorithm}")
    print(f"  Workers: {args.workers}")
    print(f"  Precompute successors: {args.precompute}")
    print(f"  Debug verification: {args.debug_verify}")
    
    if args.debug_verify:
        print(f"\n⚠️  WARNING: Debug verification enabled!")
        print(f"  This will compare every successor query with linear search.")
        print(f"  Execution will be MUCH slower but verifies correctness.")
    
    # Run the selected algorithm
    if args.algorithm == 1:
        run_algorithm_1(args)
    else:
        run_algorithm_2(args)


if __name__ == "__main__":
    main()
    
    # Example usage:
    # python safety_value_function_5.py --algorithm 2 --initial-resolution 1 --gamma 0.05 --dt 0.05 --debug-verify
    # 
    # After verifying correctness, run without debug for full speed:
    # python safety_value_function_5.py --algorithm 2 --initial-resolution 1 --gamma 0.05 --dt 0.05