"""
Complete implementation of Safety Value Functions with Formal Guarantees
All code in one file for easy copying

To run:
    python this_file.py --algorithm 1 --resolution 10 --iterations 100
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import argparse
import time
import os
from itertools import product


# ============================================================================
# ENVIRONMENT
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
    """Dubins car with constant velocity and static circular obstacle."""
    
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
        self.L_f = v_const
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
# CELL STRUCTURE
# ============================================================================

class Cell:
    """Represents a hyperrectangular cell in the discretized state space."""
    
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
    
    def get_leaves(self) -> List[Cell]:
        return self.leaves
    
    def refine_cell(self, cell: Cell):
        if not cell.is_leaf:
            return
        child1, child2 = cell.split(self.next_id)
        self.next_id += 2
        self.leaves.remove(cell)
        self.leaves.extend([child1, child2])
        cell.is_refined = True
    
    def get_cell_containing_point(self, point: np.ndarray) -> Optional[Cell]:
        for cell in self.leaves:
            if cell.contains_point(point):
                return cell
        return None
    
    def get_num_leaves(self) -> int:
        return len(self.leaves)


# ============================================================================
# REACHABILITY
# ============================================================================

class ReachabilityAnalyzer:
    """Computes forward reachable sets using grid-based over-approximation."""
    
    def __init__(self, env: Environment, samples_per_dim: int = 5):
        self.env = env
        self.samples_per_dim = samples_per_dim
    
    def compute_reachable_set(self, cell: Cell, action) -> np.ndarray:
        samples = self._sample_cell(cell)
        next_states = []
        for state in samples:
            next_state = self.env.dynamics(state, action)
            next_states.append(next_state)
        next_states = np.array(next_states)
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        reach_bounds[:, 0] = np.min(next_states, axis=0)
        reach_bounds[:, 1] = np.max(next_states, axis=0)
        return reach_bounds
    
    def compute_successor_cells(self, cell: Cell, action, cell_tree: CellTree) -> List[Cell]:
        reach_bounds = self.compute_reachable_set(cell, action)
        successors = []
        for candidate in cell_tree.get_leaves():
            if candidate.intersects(reach_bounds):
                successors.append(candidate)
        return successors
    
    def _sample_cell(self, cell: Cell) -> np.ndarray:
        dim_samples = []
        for j in range(len(cell.bounds)):
            a, b = cell.bounds[j]
            dim_samples.append(np.linspace(a, b, self.samples_per_dim))
        samples = []
        for point in product(*dim_samples):
            samples.append(np.array(point))
        return np.array(samples)


class LipschitzReachabilityAnalyzer(ReachabilityAnalyzer):
    """Enhanced reachability using Lipschitz bounds."""
    
    def compute_reachable_set(self, cell: Cell, action) -> np.ndarray:
        center = cell.center
        center_next = self.env.dynamics(center, action)
        L_f, _ = self.env.get_lipschitz_constants()
        eta = cell.get_max_range()
        reach_bounds = np.zeros((self.env.get_state_dim(), 2))
        expansion = L_f * eta
        for j in range(self.env.get_state_dim()):
            reach_bounds[j, 0] = center_next[j] - expansion
            reach_bounds[j, 1] = center_next[j] + expansion
        return reach_bounds


# ============================================================================
# VALUE ITERATION
# ============================================================================

class SafetyValueIterator:
    """Implements Algorithm 1: Discretization Routine."""
    
    def __init__(self, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: ReachabilityAnalyzer, output_dir: str = "./results"):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.L_f, self.L_l = env.get_lipschitz_constants()
        
        if gamma * self.L_f >= 1:
            raise ValueError(f"Contraction condition violated: γL_f = {gamma * self.L_f} >= 1")
        
        print(f"Initialized with γ={gamma}, L_f={self.L_f}, L_l={self.L_l}")
        print(f"Contraction factor: γL_f = {gamma * self.L_f}")
    
    def initialize_cells(self):
        for cell in self.cell_tree.get_leaves():
            l_center = self.env.failure_function(cell.center)
            eta = cell.get_max_range()
            cell.l_lower = l_center - self.L_l * eta
            cell.l_upper = l_center + self.L_l * eta
            cell.V_lower = cell.l_lower
            cell.V_upper = cell.l_upper
    
    def bellman_update(self, cell: Cell) -> Tuple[float, float]:
        max_upper = -np.inf
        max_lower = -np.inf
        
        for action in self.env.get_action_space():
            successors = self.reachability.compute_successor_cells(cell, action, self.cell_tree)
            if len(successors) == 0:
                continue
            
            upper_vals = [s.V_upper for s in successors if s.V_upper is not None]
            if upper_vals:
                action_upper = self.gamma * max(upper_vals)
                max_upper = max(max_upper, action_upper)
            
            lower_vals = [s.V_lower for s in successors if s.V_lower is not None]
            if lower_vals:
                action_lower = self.gamma * min(lower_vals)
                max_lower = max(max_lower, action_lower)
        
        new_V_upper = min(cell.l_upper, max_upper) if max_upper > -np.inf else cell.l_upper
        new_V_lower = min(cell.l_lower, max_lower) if max_lower > -np.inf else cell.l_lower
        return new_V_upper, new_V_lower
    
    def value_iteration(self, max_iterations: int = 1000, convergence_tol: float = 1e-3,
                        plot_freq: int = 10) -> Tuple[np.ndarray, np.ndarray]:
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
            
            for cell in self.cell_tree.get_leaves():
                cell.V_upper, cell.V_lower = updates[cell.cell_id]
            
            diff_upper = max(abs(cell.V_upper - prev_upper[cell.cell_id]) 
                           for cell in self.cell_tree.get_leaves())
            diff_lower = max(abs(cell.V_lower - prev_lower[cell.cell_id]) 
                           for cell in self.cell_tree.get_leaves())
            
            conv_history_upper.append(diff_upper)
            conv_history_lower.append(diff_lower)
            
            print(f"Iteration {iteration + 1}: ||V̄^k - V̄^{{k-1}}||_∞ = {diff_upper:.6f}, "
                  f"||V_^k - V_^{{k-1}}||_∞ = {diff_lower:.6f}")
            
            if (iteration + 1) % plot_freq == 0:
                self._save_plot(iteration + 1)
            
            if diff_upper < convergence_tol and diff_lower < convergence_tol:
                print(f"\nConverged at iteration {iteration + 1}!")
                self._save_plot(iteration + 1, final=True)
                break
        else:
            print(f"\nReached maximum iterations ({max_iterations})")
            self._save_plot(max_iterations, final=True)
        
        return np.array(conv_history_upper), np.array(conv_history_lower)
    
    def _save_plot(self, iteration: int, final: bool = False):
        suffix = "_final" if final else ""
        filename = os.path.join(self.output_dir, f"iteration_{iteration:04d}{suffix}.png")
        plot_value_function(self.env, self.cell_tree, filename, iteration)
        if final:
            print(f"Final plot saved to {filename}")


class AdaptiveRefinement:
    """Implements Algorithm 2/3: Adaptive refinement."""
    
    def __init__(self, env: Environment, gamma: float, cell_tree: CellTree,
                 reachability: ReachabilityAnalyzer, output_dir: str = "./results_adaptive"):
        self.env = env
        self.gamma = gamma
        self.cell_tree = cell_tree
        self.reachability = reachability
        self.output_dir = output_dir
        self.value_iterator = SafetyValueIterator(env, gamma, cell_tree, reachability, output_dir)
        self.L_l = env.get_lipschitz_constants()[1]
    
    def refine(self, epsilon: float, max_refinements: int = 100,
               vi_iterations_per_refinement: int = 100):
        eta_min = epsilon / (2 * self.L_l)
        print(f"\nAdaptive refinement with ε={epsilon}, η_min={eta_min}")
        
        print("\n=== Initial Value Iteration ===")
        self.value_iterator.value_iteration(max_iterations=vi_iterations_per_refinement, plot_freq=20)
        
        refinement_iter = 0
        while refinement_iter < max_refinements:
            boundary_cells = self._identify_boundary_cells()
            print(f"\n=== Refinement Iteration {refinement_iter + 1} ===")
            print(f"Boundary cells: {len(boundary_cells)}")
            
            if len(boundary_cells) == 0:
                print("No boundary cells remaining!")
                break
            
            refined_any = False
            for cell in boundary_cells:
                if cell.get_max_range() > eta_min:
                    self.cell_tree.refine_cell(cell)
                    refined_any = True
            
            if not refined_any:
                print("All boundary cells below minimum resolution!")
                break
            
            print(f"Total cells after refinement: {self.cell_tree.get_num_leaves()}")
            self.value_iterator.value_iteration(max_iterations=vi_iterations_per_refinement, plot_freq=20)
            refinement_iter += 1
        
        print(f"\n=== Refinement Complete ===")
        print(f"Final number of cells: {self.cell_tree.get_num_leaves()}")
        self._print_statistics()
    
    def _identify_boundary_cells(self) -> List[Cell]:
        boundary = []
        for cell in self.cell_tree.get_leaves():
            if cell.V_upper is not None and cell.V_lower is not None:
                if cell.V_upper > 0 and cell.V_lower < 0:
                    boundary.append(cell)
        return boundary
    
    def _print_statistics(self):
        safe_cells = 0
        unsafe_cells = 0
        boundary_cells = 0
        for cell in self.cell_tree.get_leaves():
            if cell.V_lower is not None and cell.V_upper is not None:
                if cell.V_lower > 0:
                    safe_cells += 1
                elif cell.V_upper < 0:
                    unsafe_cells += 1
                else:
                    boundary_cells += 1
        total = self.cell_tree.get_num_leaves()
        print(f"Safe cells: {safe_cells} ({100*safe_cells/total:.1f}%)")
        print(f"Unsafe cells: {unsafe_cells} ({100*unsafe_cells/total:.1f}%)")
        print(f"Boundary cells: {boundary_cells} ({100*boundary_cells/total:.1f}%)")


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_value_function(env: Environment, cell_tree: CellTree, filename: str,
                       iteration: int, theta_slices: list = None):
    if theta_slices is None:
        theta_slices = [0, np.pi/4, np.pi/2]
    
    n_slices = len(theta_slices)
    fig, axes = plt.subplots(2, n_slices, figsize=(5*n_slices, 10))
    if n_slices == 1:
        axes = axes.reshape(2, 1)
    
    for idx, theta in enumerate(theta_slices):
        ax_upper = axes[0, idx]
        _plot_slice(env, cell_tree, theta, ax_upper, "V̄_γ", upper=True)
        ax_upper.set_title(f"Upper Bound V̄_γ (θ={theta:.2f} rad)")
        
        ax_lower = axes[1, idx]
        _plot_slice(env, cell_tree, theta, ax_lower, "V_γ", upper=False)
        ax_lower.set_title(f"Lower Bound V_γ (θ={theta:.2f} rad)")
    
    fig.suptitle(f"Safety Value Function - Iteration {iteration}", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_slice(env: Environment, cell_tree: CellTree, theta: float, ax, label: str, upper: bool):
    bounds = env.get_state_bounds()
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # 1. Plot the value function as before
    resolution = 100
    x_grid = np.linspace(x_min, x_max, resolution)
    y_grid = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

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

    # 2. Filled contour for smooth background
    im = ax.contourf(X, Y, V, levels=20, cmap='RdYlGn', alpha=0.75)

    # 3. Draw the 0-level set (black thick line)
    ax.contour(X, Y, V, levels=[0.0], colors='black', linewidths=2.5)

    # 4. Draw black grid borders around each leaf cell
    for cell in cell_tree.get_leaves():
        a_x, b_x = cell.bounds[0]
        a_y, b_y = cell.bounds[1]

        rect = Rectangle(
            (a_x, a_y),
            b_x - a_x,
            b_y - a_y,
            fill=False,
            edgecolor='black',
            linewidth=0.5,       # you can make thicker if needed
            alpha=0.8,
        )
        ax.add_patch(rect)

    # 5. Draw obstacle (optional)
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          color='red', alpha=0.5, zorder=10)
        ax.add_patch(obstacle)

    # 6. Aesthetic setup
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    plt.colorbar(im, ax=ax).set_label(label)



def plot_convergence(conv_upper: np.ndarray, conv_lower: np.ndarray, filename: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = np.arange(1, len(conv_upper) + 1)
    ax.semilogy(iterations, conv_upper, 'b-', label='||V̄^k - V̄^{k-1}||_∞', linewidth=2)
    ax.semilogy(iterations, conv_lower, 'r-', label='||V_^k - V_^{k-1}||_∞', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Infinity Norm Difference', fontsize=12)
    ax.set_title('Value Iteration Convergence', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Convergence plot saved to {filename}")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np

def plot_reachability_single_cell(env, cell_tree, reachability, cell_idx=None,
                                  n_samples=50, figsize=(7,7)):
    """
    Visualize reachable cells from a single source cell for each action,
    with colored arrows for reachability and small vectors showing true one-step dynamics
    from sampled points within the source cell.
    """
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

    # Colors
    action_colors = {-1.0: 'blue', 0.0: 'green', 1.0: 'orange'}
    offset_angles = {-1.0: -8*np.pi/180, 0.0: 0.0, 1.0: 8*np.pi/180}

    # Draw grid
    for cell in leaves:
        rect = Rectangle((cell.bounds[0,0], cell.bounds[1,0]),
                         cell.get_range(0), cell.get_range(1),
                         fill=False, edgecolor='gray', linewidth=0.4, alpha=0.3)
        ax.add_patch(rect)

    # Highlight source cell
    rect_src = Rectangle((src_cell.bounds[0,0], src_cell.bounds[1,0]),
                         src_cell.get_range(0), src_cell.get_range(1),
                         fill=False, edgecolor='black', linewidth=2.0)
    ax.add_patch(rect_src)
    ax.plot(*src_center, 'ko', markersize=6, label='Source cell center')

    # (1) Grid-level reachability (colored large arrows)
    for action in env.get_action_space():
        succ_cells = reachability.compute_successor_cells(src_cell, action, cell_tree)
        color = action_colors[action]
        offset_angle = offset_angles[action]

        for dst_cell in succ_cells:
            dst_center = dst_cell.center[:2]
            dx = dst_center[0] - src_center[0]
            dy = dst_center[1] - src_center[1]

            if offset_angle != 0.0:
                rot = np.array([[np.cos(offset_angle), -np.sin(offset_angle)],
                                [np.sin(offset_angle),  np.cos(offset_angle)]])
                dx, dy = rot @ np.array([dx, dy])

            ax.arrow(src_center[0], src_center[1], dx, dy,
                     color=color, alpha=0.6, linewidth=2.0,
                     head_width=0.10, length_includes_head=True)

        ax.plot([], [], color=color, linewidth=2.0, label=f"Action {action:+.0f}")

    # (2) Local sample-based visualization (short motion vectors)
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.uniform(src_cell.bounds[0,0], src_cell.bounds[0,1], n_samples),
        rng.uniform(src_cell.bounds[1,0], src_cell.bounds[1,1], n_samples),
        rng.uniform(src_cell.bounds[2,0], src_cell.bounds[2,1], n_samples)
    ])

    for s in samples:
        x0, y0 = s[:2]
        for action in env.get_action_space():##doesnt matter since action doesn't affect delta x and delta y using our discretized dynamics
            color = action_colors[action]
            next_s = env.dynamics(s, action)
            dx = next_s[0] - x0
            dy = next_s[1] - y0

            # scale vector for visibility — relative to motion magnitude
            scale = 1.0  # can tweak (e.g., 1.0 or 0.5)
            ax.plot([x0, x0 + scale*dx], [y0, y0 + scale*dy],
                    color=color, alpha=0.9, linewidth=1.8, zorder=5)

        # mark start point
        ax.plot(x0, y0, 'ko', markersize=3, alpha=0.7)

    # Obstacle
    if isinstance(env, DubinsCarEnvironment):
        obstacle = Circle(env.obstacle_position, env.obstacle_radius,
                          color='red', alpha=0.5, zorder=4)
        ax.add_patch(obstacle)

    ax.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN INTERFACE
# ============================================================================

def run_algorithm_1(args):
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
    
    if args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using Lipschitz-based reachability")
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
        print(f"  Using sampling-based reachability ({args.samples} samples/dim)")
    plot_reachability_single_cell(env, cell_tree, reachability, cell_idx=45)

    value_iter = SafetyValueIterator(env=env, gamma=args.gamma, cell_tree=cell_tree,
                                     reachability=reachability, output_dir=args.output_dir)
    
    start_time = time.time()
    conv_upper, conv_lower = value_iter.value_iteration(max_iterations=args.iterations,
                                                        convergence_tol=args.tolerance,
                                                        plot_freq=args.plot_freq)
    elapsed = time.time() - start_time
    
    print(f"\nTotal computation time: {elapsed:.2f} seconds")
    print(f"Time per iteration: {elapsed/len(conv_upper):.3f} seconds")
    
    plot_convergence(conv_upper, conv_lower, f"{args.output_dir}/convergence.png")
    print(f"\nResults saved to: {args.output_dir}/")


def run_algorithm_2(args):
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
    
    if args.lipschitz_reach:
        reachability = LipschitzReachabilityAnalyzer(env, samples_per_dim=args.samples)
    else:
        reachability = ReachabilityAnalyzer(env, samples_per_dim=args.samples)
    
    adaptive = AdaptiveRefinement(env=env, gamma=args.gamma, cell_tree=cell_tree,
                                  reachability=reachability, output_dir=args.output_dir)
    
    start_time = time.time()
    adaptive.refine(epsilon=args.epsilon, max_refinements=args.refinements,
                   vi_iterations_per_refinement=args.vi_iterations)
    elapsed = time.time() - start_time
    
    print(f"\nTotal computation time: {elapsed:.2f} seconds")
    print(f"\nResults saved to: {args.output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Safety Value Function Computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--algorithm', type=int, choices=[1, 2], required=True,
                       help="Algorithm to run: 1 (basic) or 2 (adaptive)")
    parser.add_argument('--velocity', type=float, default=1.0, help="Constant velocity")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step")
    parser.add_argument('--obstacle-radius', type=float, default=0.5, help="Obstacle radius")
    parser.add_argument('--gamma', type=float, default=0.95, help="Discount factor")
    
    # Algorithm 1
    parser.add_argument('--resolution', type=int, default=10, help="Grid resolution")
    parser.add_argument('--iterations', type=int, default=200, help="Max iterations")
    parser.add_argument('--tolerance', type=float, default=1e-3, help="Convergence tolerance")
    parser.add_argument('--plot-freq', type=int, default=1, help="Plot frequency")
    
    # Algorithm 2
    parser.add_argument('--epsilon', type=float, default=0.1, help="Error tolerance")
    parser.add_argument('--initial-resolution', type=int, default=8, help="Initial resolution")
    parser.add_argument('--refinements', type=int, default=5, help="Max refinements")
    parser.add_argument('--vi-iterations', type=int, default=100, help="VI iterations per refinement")
    
    # Reachability
    parser.add_argument('--samples', type=int, default=10, help="Samples per dimension")
    parser.add_argument('--lipschitz-reach', action='store_true', help="Use Lipschitz reachability")
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./results', help="Output directory")
    
    args = parser.parse_args()
    
    if args.algorithm == 1:
        run_algorithm_1(args)
    else:
        run_algorithm_2(args)


if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
#              ↑ θ = π/2 (90)
#  θ = π (180) ←----o----→  θ = 0
#              |
#              ↓ θ = -π/2 (-90)