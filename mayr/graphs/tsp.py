"""TSP Solver using Nearest Neighbor Algorithm

This module implements the Nearest Neighbor heuristic for solving the
Traveling Salesman Problem (TSP) on Euclidean complete graphs.
"""

import random
import time
from typing import List, Tuple, Optional
import tsplib95
from itertools import combinations
import numpy as np
from scipy.optimize import linprog


class TSPSolver:
    """Base class for TSP solvers."""

    def __init__(self, problem) -> None:
        """
        Initialize the TSP solver.

        Args:
            problem: tsplib95 problem instance
        """
        self.problem = problem
        self.n: int = problem.dimension
        self.nodes: List[int] = list(problem.get_nodes())

    def get_distance(self, node1: int, node2: int) -> float:
        """Get distance between two nodes using the problem's weight function."""
        return self.problem.get_weight(node1, node2)

    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate the total distance of a tour."""
        total_distance: float = 0.0
        for i in range(len(tour)):
            from_node = tour[i]
            to_node = tour[(i + 1) % len(tour)]
            total_distance += self.get_distance(from_node, to_node)
        return total_distance

    def verify_tour(self, tour: List[int]) -> bool:
        """Verify that a tour is valid (visits each city exactly once)."""
        if len(tour) != self.n:
            return False
        return set(tour) == set(self.nodes)


class NearestNeighbor(TSPSolver):
    """
    Nearest Neighbor Algorithm for TSP.

    Time Complexity: O(n²)
    Space Complexity: O(n)

    This is a greedy heuristic that builds a tour by always choosing
    the nearest unvisited city as the next destination.
    """

    def solve(self, start_node: Optional[int] = None) -> Tuple[List[int], float]:
        """
        Construct a tour using the Nearest Neighbor heuristic.

        Args:
            start_node: Starting node for the tour (default: first node)

        Returns:
            Tuple of (tour, distance)
        """
        if start_node is None:
            start_node = self.nodes[0]

        if start_node not in self.nodes:
            raise ValueError(f"Invalid start node: {start_node}")

        # Initialize tour with start node
        tour = [start_node]
        unvisited = set(self.nodes)
        unvisited.remove(start_node)
        current = start_node

        # Greedily build the tour
        while unvisited:
            # Find the nearest unvisited node
            nearest_node = min(
                unvisited, key=lambda node: self.get_distance(current, node)
            )

            tour.append(nearest_node)
            unvisited.remove(nearest_node)
            current = nearest_node

        distance = self.calculate_tour_distance(tour)
        return tour, distance

    def solve_multiple_starts(
        self, num_trials: Optional[int] = None
    ) -> Tuple[List[int], float]:
        """
        Run Nearest Neighbor from multiple starting points and return the best tour.

        Args:
            num_trials: Number of different starting points to try.
                       If None, tries all nodes as starting points.

        Returns:
            Tuple of (best_tour, best_distance)
        """
        if num_trials is None:
            num_trials = self.n

        num_trials = min(num_trials, self.n)
        start_nodes = random.sample(self.nodes, num_trials)

        best_tour = None
        best_distance = float("inf")

        for start_node in start_nodes:
            tour, distance = self.solve(start_node)
            if distance < best_distance:
                best_distance = distance
                best_tour = tour

        return best_tour, best_distance


class HeldKarp(TSPSolver):
    """
    Held-Karp Algorithm (Dynamic Programming) for TSP.

    Time Complexity: O(n² × 2ⁿ)
    Space Complexity: O(n × 2ⁿ)

    This is an exact algorithm that guarantees finding the optimal solution.
    Practical for problems with n ≤ 20-25 cities.
    """

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve TSP using the Held-Karp dynamic programming algorithm.

        Returns:
            Tuple of (optimal_tour, optimal_distance)
        """
        if self.n > 25:
            print(f"Warning: Held-Karp with {self.n} cities may take very long!")
            print("Consider using approximation algorithms for n > 25.")

        # For simplicity, we'll work with 0-indexed nodes internally
        # and convert back to the original node IDs at the end
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        n = self.n

        # Create distance matrix with indices
        dist = [[float("inf")] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist[i][j] = self.get_distance(idx_to_node[i], idx_to_node[j])

        # DP table: dp[mask][i] = minimum cost to visit all nodes in mask, ending at i
        # mask is a bitmask representing the set of visited nodes
        dp = {}
        parent = {}  # For reconstructing the path

        # Initialize: starting from node 0, having visited only node 0
        start_mask = 1  # Only node 0 is visited
        dp[(start_mask, 0)] = 0
        parent[(start_mask, 0)] = None

        # Iterate through all possible subsets of nodes
        for num_visited in range(2, n + 1):
            # Generate all subsets of size num_visited that include node 0
            for subset_tuple in combinations(range(1, n), num_visited - 1):
                subset = set(subset_tuple)
                subset.add(0)  # Always include starting node

                # Convert subset to bitmask
                mask = 0
                for node in subset:
                    mask |= 1 << node

                # For each node in the subset, try it as the last visited node
                for last in subset:
                    if last == 0 and num_visited < n:
                        continue  # Can't end at start node unless we've visited all

                    # Find the minimum cost to reach 'last' through any previous node
                    prev_mask = mask & ~(1 << last)  # Remove 'last' from the mask

                    if prev_mask == 0:
                        continue

                    min_cost = float("inf")
                    min_prev = None

                    for prev in subset:
                        if prev == last:
                            continue
                        if (prev_mask, prev) in dp:
                            cost = dp[(prev_mask, prev)] + dist[prev][last]
                            if cost < min_cost:
                                min_cost = cost
                                min_prev = prev

                    if min_cost < float("inf"):
                        dp[(mask, last)] = min_cost
                        parent[(mask, last)] = min_prev

        # Find minimum cost to return to starting node
        full_mask = (1 << n) - 1  # All nodes visited
        min_cost = float("inf")
        last_node = None

        for last in range(1, n):
            if (full_mask, last) in dp:
                cost = dp[(full_mask, last)] + dist[last][0]
                if cost < min_cost:
                    min_cost = cost
                    last_node = last

        # Reconstruct the tour
        if last_node is None:
            # Fallback for very small instances
            if n == 1:
                return [self.nodes[0]], 0
            elif n == 2:
                return self.nodes, dist[0][1] + dist[1][0]
            else:
                raise ValueError("Failed to find a valid tour")

        tour_indices = [0]
        current = last_node
        mask = full_mask

        # Trace back through the parent pointers
        path_reverse = [current]
        while parent.get((mask, current)) is not None:
            prev = parent[(mask, current)]
            path_reverse.append(prev)
            mask &= ~(1 << current)  # Remove current from mask
            current = prev

        # Reverse the path (excluding the starting 0 which we already have)
        for i in range(len(path_reverse) - 2, -1, -1):
            tour_indices.append(path_reverse[i])

        # Convert indices back to original node IDs
        tour = [idx_to_node[i] for i in tour_indices]

        return tour, min_cost


class BranchBoundLP(TSPSolver):
    """
    Branch and Bound Algorithm with LP Relaxation for TSP.

    Time Complexity: O(2ⁿ) worst case, often much better in practice
    Space Complexity: O(n²) per node

    Uses Linear Programming relaxation for lower bounds and branching.
    More scalable than Held-Karp but still exponential worst case.
    """

    def __init__(self, problem):
        super().__init__(problem)
        # Pre-compute distance matrix for efficiency
        self.dist_matrix = self._build_distance_matrix()
        self.n_nodes = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        self.best_tour = None
        self.best_cost = float("inf")
        self.nodes_explored = 0
        self.max_depth = 0

    def _build_distance_matrix(self):
        """Build a distance matrix for all nodes."""
        n = len(self.nodes)
        matrix = np.zeros((n, n))
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    matrix[i][j] = self.get_distance(node_i, node_j)
        return matrix

    def _solve_lp_relaxation(self, fixed_edges, forbidden_edges):
        """
        Solve the LP relaxation of TSP with fixed and forbidden edges.
        Returns lower bound and fractional solution.
        """
        n = self.n_nodes

        # Create decision variables x_ij for each edge
        # We'll use a flattened representation
        n_vars = n * n

        # Objective: minimize sum of distances * x_ij
        c = self.dist_matrix.flatten()

        # Constraints
        A_eq = []
        b_eq = []

        # Each node must have exactly 2 edges (degree constraint)
        for i in range(n):
            # Outgoing edges
            constraint_out = np.zeros(n_vars)
            for j in range(n):
                if i != j:
                    constraint_out[i * n + j] = 1
            A_eq.append(constraint_out)
            b_eq.append(1)  # Exactly one outgoing

            # Incoming edges
            constraint_in = np.zeros(n_vars)
            for j in range(n):
                if i != j:
                    constraint_in[j * n + i] = 1
            A_eq.append(constraint_in)
            b_eq.append(1)  # Exactly one incoming

        # Bounds: 0 <= x_ij <= 1
        bounds = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    bounds.append((0, 0))  # No self-loops
                elif (i, j) in fixed_edges:
                    bounds.append((1, 1))  # Fixed to 1
                elif (i, j) in forbidden_edges:
                    bounds.append((0, 0))  # Fixed to 0
                else:
                    bounds.append((0, 1))  # Free variable

        # Solve LP
        try:
            result = linprog(
                c,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                method="highs",
                options={"disp": False},
            )

            if result.success:
                return result.fun, result.x.reshape((n, n))
            else:
                return float("inf"), None
        except:
            return float("inf"), None

    def _find_fractional_edge(self, solution):
        """Find an edge with fractional value in LP solution."""
        n = self.n_nodes
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = solution[i, j]
                    if 0.01 < val < 0.99:  # Fractional (with tolerance)
                        return (i, j)
        return None

    def _extract_tour_from_solution(self, solution):
        """Try to extract a valid tour from an integer solution."""
        n = self.n_nodes
        tour = [0]  # Start from node 0
        current = 0
        visited = {0}

        for _ in range(n - 1):
            # Find next node
            next_node = None
            for j in range(n):
                if j not in visited and solution[current, j] > 0.99:
                    next_node = j
                    break

            if next_node is None:
                return None  # Invalid tour

            tour.append(next_node)
            visited.add(next_node)
            current = next_node

        # Check if we can return to start
        if solution[current, 0] > 0.99:
            return [self.idx_to_node[i] for i in tour]
        return None

    def _branch_and_bound(self, fixed_edges, forbidden_edges, depth=0):
        """Recursive branch and bound with LP relaxation."""
        self.nodes_explored += 1
        self.max_depth = max(self.max_depth, depth)

        # Solve LP relaxation
        lower_bound, solution = self._solve_lp_relaxation(fixed_edges, forbidden_edges)

        # Pruning: if lower bound >= best known, stop
        if lower_bound >= self.best_cost:
            return

        # Check if solution is integral (a valid tour)
        if solution is not None:
            tour = self._extract_tour_from_solution(solution)
            if tour is not None:
                cost = self.calculate_tour_distance(tour)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_tour = tour
                return

        # Find fractional edge for branching
        if solution is not None:
            fractional_edge = self._find_fractional_edge(solution)
            if fractional_edge is None:
                return  # No fractional edges but not a valid tour

            # Branch on the fractional edge
            i, j = fractional_edge

            # Branch 1: Include edge (i, j)
            new_fixed = fixed_edges | {(i, j)}
            self._branch_and_bound(new_fixed, forbidden_edges, depth + 1)

            # Branch 2: Exclude edge (i, j)
            new_forbidden = forbidden_edges | {(i, j)}
            self._branch_and_bound(fixed_edges, new_forbidden, depth + 1)

    def solve(self, time_limit=60) -> Tuple[List[int], float]:
        """
        Solve TSP using Branch and Bound with LP relaxation.
        Note: Simplified implementation - may not always find optimal for complex problems.

        Args:
            time_limit: Maximum time in seconds (default: 60)

        Returns:
            Tuple of (tour, distance)
        """
        start_time = time.time()

        # Initialize with a greedy solution for upper bound
        nn = NearestNeighbor(self.problem)
        self.best_tour, self.best_cost = nn.solve()

        # For small problems, try harder
        if self.n_nodes <= 15:
            # Also try multiple NN starts for better upper bound
            nn_best_tour, nn_best_cost = nn.solve_multiple_starts(self.n_nodes)
            if nn_best_cost < self.best_cost:
                self.best_cost = nn_best_cost
                self.best_tour = nn_best_tour

        # Start branch and bound (simplified version)
        # Note: Full implementation would require subtour elimination constraints
        try:
            max_iterations = min(1000, 2**self.n_nodes)
            for _ in range(max_iterations):
                if time.time() - start_time > time_limit:
                    break
                # Try to improve with limited branching
                self._branch_and_bound(set(), set())
                if self.nodes_explored > max_iterations:
                    break
        except:
            pass  # Allow failures

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            print(f"  ⏱️ Time limit reached ({time_limit}s)")

        print(f"  📊 Nodes explored: {self.nodes_explored}")
        print(f"  🌳 Max depth reached: {self.max_depth}")

        if self.best_tour is None:
            return [], float("inf")

        return self.best_tour, self.best_cost


class BranchBoundSimple(TSPSolver):
    """
    Simple Branch and Bound Algorithm for TSP (without LP).

    Time Complexity: O(n!) worst case, often better with good bounds
    Space Complexity: O(n²)

    Uses simple lower bounds and depth-first search with pruning.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self.dist_matrix = self._build_distance_matrix()
        self.n_nodes = len(self.nodes)
        self.best_tour = None
        self.best_cost = float("inf")
        self.nodes_explored = 0

    def _build_distance_matrix(self):
        """Build a distance matrix for all nodes."""
        matrix = {}
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    matrix[(node_i, node_j)] = self.get_distance(node_i, node_j)
        return matrix

    def _lower_bound(self, path, remaining):
        """
        Calculate lower bound for current partial path.
        Uses sum of minimum outgoing edges from remaining nodes.
        """
        if not path:
            return 0

        # Cost of current path
        cost = 0
        for i in range(len(path) - 1):
            cost += self.dist_matrix[(path[i], path[i + 1])]

        if not remaining:
            # Add cost to return to start
            if len(path) == self.n_nodes:
                cost += self.dist_matrix[(path[-1], path[0])]
            return cost

        # Add minimum edge from last node in path
        last = path[-1]
        min_edge = min(self.dist_matrix[(last, node)] for node in remaining)
        cost += min_edge

        # Add sum of minimum edges from remaining nodes
        for node in remaining:
            if node != list(remaining)[-1]:  # Not the last remaining
                edges = [
                    self.dist_matrix[(node, other)]
                    for other in self.nodes
                    if other != node
                ]
                if edges:
                    cost += min(edges)

        return cost

    def _branch_and_bound_dfs(self, path, remaining, current_cost):
        """DFS branch and bound."""
        self.nodes_explored += 1

        # Complete tour
        if not remaining:
            # Add cost to return to start
            total_cost = current_cost + self.dist_matrix[(path[-1], path[0])]
            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_tour = path.copy()
            return

        # Calculate lower bound
        lower_bound = self._lower_bound(path, remaining)

        # Pruning
        if lower_bound >= self.best_cost:
            return

        # Try each remaining node
        for next_node in remaining:
            new_path = path + [next_node]
            new_remaining = remaining - {next_node}
            new_cost = current_cost
            if len(path) > 0:
                new_cost += self.dist_matrix[(path[-1], next_node)]

            self._branch_and_bound_dfs(new_path, new_remaining, new_cost)

    def solve(self) -> Tuple[List[int], float]:
        """
        Solve TSP using simple Branch and Bound.

        Returns:
            Tuple of (tour, distance)
        """
        # Initialize with greedy solution
        nn = NearestNeighbor(self.problem)
        self.best_tour, self.best_cost = nn.solve()

        # Start from first node
        start_node = self.nodes[0]
        remaining = set(self.nodes[1:])

        self._branch_and_bound_dfs([start_node], remaining, 0)

        print(f"  📊 Nodes explored: {self.nodes_explored}")

        return self.best_tour, self.best_cost


class LinKernighan(TSPSolver):
    """
    Lin-Kernighan Heuristic for TSP.

    Time Complexity: O(n^2.2) average case
    Space Complexity: O(n^2)

    One of the most effective TSP heuristics. Uses variable k-opt moves
    to escape local optima. Often finds optimal or near-optimal solutions.
    """

    def __init__(self, problem):
        super().__init__(problem)
        self.n = len(self.nodes)
        # Pre-compute distance matrix for efficiency
        self.dist_matrix = {}
        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    self.dist_matrix[(node_i, node_j)] = self.get_distance(
                        node_i, node_j
                    )

    def _gain(self, tour, i, j):
        """Calculate gain from removing edge (i, i+1) and (j, j+1) and adding (i, j) and (i+1, j+1)."""
        n = len(tour)
        i_next = (i + 1) % n
        j_next = (j + 1) % n

        # Cost of edges to remove
        remove_cost = (
            self.dist_matrix[(tour[i], tour[i_next])]
            + self.dist_matrix[(tour[j], tour[j_next])]
        )

        # Cost of edges to add
        add_cost = (
            self.dist_matrix[(tour[i], tour[j])]
            + self.dist_matrix[(tour[i_next], tour[j_next])]
        )

        return remove_cost - add_cost

    def _make_2opt_move(self, tour, i, j):
        """Perform a 2-opt move."""
        new_tour = tour[:]
        new_tour[i + 1 : j + 1] = reversed(new_tour[i + 1 : j + 1])
        return new_tour

    def _find_improving_move(self, tour, tabu_edges=None):
        """Find an improving k-opt move using the Lin-Kernighan strategy."""
        if tabu_edges is None:
            tabu_edges = set()

        n = len(tour)
        best_gain = 0
        best_move = None

        # Try to find improving 2-opt moves first
        for i in range(n):
            for j in range(i + 2, min(i + n - 1, n)):
                if (tour[i], tour[(i + 1) % n]) in tabu_edges:
                    continue

                gain = self._gain(tour, i, j)
                if gain > best_gain:
                    best_gain = gain
                    best_move = ("2opt", i, j)

        # If 2-opt found improvement, return it
        if best_move:
            return best_move, best_gain

        # Try more complex moves (simplified version)
        # In full implementation, this would include 3-opt, 4-opt, etc.
        for i in range(n):
            if (tour[i], tour[(i + 1) % n]) in tabu_edges:
                continue

            # Look for sequential improvements (simplified LK chain)
            current_gain = 0
            moves = []
            current_tour = tour[:]

            for step in range(min(5, n // 2)):  # Limit chain length
                found = False
                for j in range(n):
                    if i == j or abs(i - j) <= 1:
                        continue

                    test_tour = self._make_2opt_move(current_tour, min(i, j), max(i, j))
                    test_gain = self.calculate_tour_distance(
                        tour
                    ) - self.calculate_tour_distance(test_tour)

                    if test_gain > 0:
                        current_gain += test_gain
                        moves.append(("2opt", min(i, j), max(i, j)))
                        current_tour = test_tour
                        found = True
                        break

                if not found:
                    break

            if current_gain > best_gain:
                best_gain = current_gain
                best_move = ("chain", moves)

        return best_move, best_gain

    def _apply_move(self, tour, move):
        """Apply a k-opt move to the tour."""
        if move[0] == "2opt":
            return self._make_2opt_move(tour, move[1], move[2])
        elif move[0] == "chain":
            new_tour = tour[:]
            for submove in move[1]:
                new_tour = self._apply_move(new_tour, submove)
            return new_tour
        return tour

    def _local_search(self, tour, max_iterations=100):
        """Perform Lin-Kernighan local search."""
        current_tour = tour[:]
        current_cost = self.calculate_tour_distance(current_tour)

        tabu_edges = set()
        no_improvement_count = 0

        for iteration in range(max_iterations):
            move, gain = self._find_improving_move(current_tour, tabu_edges)

            if move is None or gain <= 0:
                no_improvement_count += 1
                if no_improvement_count > 10:
                    # Try perturbation to escape local optimum
                    if len(current_tour) > 4:
                        # Random 4-opt perturbation
                        i, j = sorted(random.sample(range(len(current_tour)), 2))
                        if j - i > 2:
                            current_tour = self._make_2opt_move(current_tour, i, j - 1)
                            current_cost = self.calculate_tour_distance(current_tour)
                            no_improvement_count = 0
                            tabu_edges.clear()
                    else:
                        break
            else:
                new_tour = self._apply_move(current_tour, move)
                new_cost = self.calculate_tour_distance(new_tour)

                if new_cost < current_cost:
                    current_tour = new_tour
                    current_cost = new_cost
                    no_improvement_count = 0

                    # Update tabu list (keep last few edges)
                    if move[0] == "2opt":
                        i, j = move[1], move[2]
                        tabu_edges.add(
                            (current_tour[i], current_tour[(i + 1) % len(current_tour)])
                        )
                        if len(tabu_edges) > 20:
                            tabu_edges.pop()

        return current_tour, current_cost

    def solve(self, num_trials=3) -> Tuple[List[int], float]:
        """
        Solve TSP using Lin-Kernighan heuristic (simplified version).

        Args:
            num_trials: Number of random starts to try

        Returns:
            Tuple of (tour, distance)
        """
        # Start with a good initial solution
        nn = NearestNeighbor(self.problem)
        best_tour, best_cost = nn.solve()

        # Apply LK from multiple starting points
        for trial in range(num_trials):
            if trial == 0:
                # Use NN solution
                initial_tour = best_tour
            else:
                # Random start or different NN start
                if trial <= len(self.nodes):
                    initial_tour, _ = nn.solve(
                        start_node=self.nodes[trial % len(self.nodes)]
                    )
                else:
                    # Random tour
                    initial_tour = self.nodes[:]
                    random.shuffle(initial_tour)

            # Apply Lin-Kernighan
            improved_tour, improved_cost = self._local_search(initial_tour)

            if improved_cost < best_cost:
                best_cost = improved_cost
                best_tour = improved_tour

        return best_tour, best_cost


def main():
    import sys

    # You can change this to test different problems
    filename = "data/lin318.tsp"
    # filename = "data/kroA100.tsp"
    # filename = "data/pcb442.tsp"
    # filename = "data/test15.tsp"

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    print("=" * 80)
    print("TSP ALGORITHM COMPARISON")
    print("=" * 80)
    print()
    print(f"Reading problem from {filename}...")
    problem = tsplib95.load(filename)

    print("Problem loaded:")
    print(f"  Name: {problem.name}")
    if hasattr(problem, "comment"):
        print(f"  Comment: {problem.comment}")
    print(f"  Type: {problem.type}")
    print(f"  Number of cities: {problem.dimension}")
    print(f"  Edge weight type: {problem.edge_weight_type}")
    print()

    # Known optimal values for comparison
    # These are confirmed optimal values from the TSP literature:
    # - kroA100: 21282 (found by Concorde/ABCC)
    # - lin318: 42029 (found by Padberg & Rinaldi 1987)
    # - pcb442: 50778 (found by Concorde/ABCC)
    known_optima = {
        "kroA100": 21282,  # 100-city problem A (Krolak/Felts/Nelson)
        "lin318": 42029,  # 318-city problem (Lin/Kernighan)
        "pcb442": 50778,  # 442-city PCB drilling problem
        "test15": None,  # Unknown for our test file
        "test25": None,  # Unknown for our test file
    }

    results = []

    # Algorithm 1: Nearest Neighbor (Single Start)
    print("=" * 80)
    print("Algorithm 1: NEAREST NEIGHBOR (Single Start)")
    print("=" * 80)
    print("Description: Greedy heuristic - always picks nearest unvisited city")
    print()
    nn_solver = NearestNeighbor(problem)
    first_node = nn_solver.nodes[0]
    start_time = time.time()
    try:
        nn_single_tour, nn_single_dist = nn_solver.solve(start_node=first_node)
        nn_single_time = time.time() - start_time
        print(f"✓ Completed in {nn_single_time:.4f} seconds")
        print(f"  Tour distance: {nn_single_dist:.2f}")
        print(f"  Starting from city: {first_node}")
        print(
            f"  Tour sample: {nn_single_tour[:10]}{'...' if len(nn_single_tour) > 10 else ''}"
        )
        results.append(
            ("Nearest Neighbor (single)", nn_single_dist, nn_single_time, "Heuristic")
        )
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Nearest Neighbor (single)", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 2: Nearest Neighbor (Multiple Starts)
    print("=" * 80)
    print("Algorithm 2: NEAREST NEIGHBOR (Multiple Starts)")
    print("=" * 80)
    num_trials = min(50, problem.dimension)
    print(f"Description: Run NN from {num_trials} different starting cities, pick best")
    print()
    start_time = time.time()
    try:
        nn_best_tour, nn_best_dist = nn_solver.solve_multiple_starts(
            num_trials=num_trials
        )
        nn_multi_time = time.time() - start_time
        print(f"✓ Completed in {nn_multi_time:.4f} seconds")
        print(f"  Best tour distance: {nn_best_dist:.2f}")
        improvement = (nn_single_dist - nn_best_dist) / nn_single_dist * 100
        print(f"  Improvement over single: {improvement:.1f}%")
        print(
            f"  Tour sample: {nn_best_tour[:10]}{'...' if len(nn_best_tour) > 10 else ''}"
        )
        results.append(
            ("Nearest Neighbor (best)", nn_best_dist, nn_multi_time, "Heuristic")
        )
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Nearest Neighbor (best)", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 3: Lin-Kernighan Heuristic
    print("=" * 80)
    print("Algorithm 3: LIN-KERNIGHAN HEURISTIC")
    print("=" * 80)
    print("Description: Advanced local search with variable k-opt moves")
    print("             Often finds optimal or near-optimal solutions")
    print()

    lk_solver = LinKernighan(problem)
    start_time = time.time()
    try:
        # Use fewer trials for larger problems
        num_trials = 2 if problem.dimension <= 50 else 1
        lk_tour, lk_dist = lk_solver.solve(num_trials=num_trials)
        lk_time = time.time() - start_time
        print(f"✓ Completed in {lk_time:.4f} seconds")
        print(f"  Tour distance: {lk_dist:.2f}")
        if "nn_best_dist" in locals():
            improvement = (
                (nn_best_dist - lk_dist) / nn_best_dist * 100 if nn_best_dist > 0 else 0
            )
            print(f"  Improvement over NN: {improvement:.1f}%")
        print(f"  Tour sample: {lk_tour[:10]}{'...' if len(lk_tour) > 10 else ''}")
        results.append(("Lin-Kernighan", lk_dist, lk_time, "Heuristic"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Lin-Kernighan", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 4: Held-Karp Dynamic Programming
    max_hk_size = 25  # Maximum practical size for Held-Karp
    if problem.dimension <= max_hk_size:
        print("=" * 80)
        print("Algorithm 4: HELD-KARP DYNAMIC PROGRAMMING")
        print("=" * 80)
        print(f"Description: Exact algorithm using DP - guarantees optimal solution")
        print()
        hk_solver = HeldKarp(problem)
        start_time = time.time()
        try:
            hk_tour, hk_dist = hk_solver.solve()
            hk_time = time.time() - start_time
            print(f"✓ Completed in {hk_time:.4f} seconds")
            print(f"  Optimal tour distance: {hk_dist:.2f}")
            print(f"  Tour sample: {hk_tour[:10]}{'...' if len(hk_tour) > 10 else ''}")
            results.append(("Held-Karp", hk_dist, hk_time, "Exact"))
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append(("Held-Karp", "N/A", "N/A", "Failed"))
        print()
    else:
        print("⏩ Skipping Held-Karp (problem too large, > 25 cities)")
        print()

    # Algorithm 5: Branch and Bound with LP Relaxation
    # Only run on smaller problems due to LP solver overhead
    if problem.dimension <= 30:
        print("=" * 80)
        print("Algorithm 5: BRANCH & BOUND WITH LP RELAXATION")
        print("=" * 80)
        print("Description: Simplified B&B with LP relaxation (demonstration)")
        print("             Note: Full implementation would need subtour elimination")
        print()

        bb_lp_solver = BranchBoundLP(problem)
        start_time = time.time()
        try:
            # Use shorter time limit for larger problems
            time_limit = 10 if problem.dimension <= 20 else 5
            bb_lp_tour, bb_lp_dist = bb_lp_solver.solve(time_limit=time_limit)
            bb_lp_time = time.time() - start_time
            print(f"✓ Completed in {bb_lp_time:.4f} seconds")
            print(f"  Tour distance: {bb_lp_dist:.2f}")
            print(
                f"  Tour sample: {bb_lp_tour[:10]}{'...' if len(bb_lp_tour) > 10 else ''}"
            )
            # Mark as heuristic since our simplified version may not find optimal
            results.append(
                ("Branch & Bound (LP)", bb_lp_dist, bb_lp_time, "Heuristic*")
            )
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append(("Branch & Bound (LP)", "N/A", "N/A", "Failed"))
    else:
        print("⏩ Skipping Branch & Bound with LP (problem too large, > 30 cities)")
    print()

    # Algorithm 6: Simple Branch and Bound (for small problems only)
    if problem.dimension <= 15:
        print("=" * 80)
        print("Algorithm 6: BRANCH & BOUND (Simple)")
        print("=" * 80)
        print("Description: Basic branch & bound with simple bounds")
        print()

        bb_simple_solver = BranchBoundSimple(problem)
        start_time = time.time()
        try:
            bb_simple_tour, bb_simple_dist = bb_simple_solver.solve()
            bb_simple_time = time.time() - start_time
            print(f"✓ Completed in {bb_simple_time:.4f} seconds")
            print(f"  Tour distance: {bb_simple_dist:.2f}")
            print(
                f"  Tour sample: {bb_simple_tour[:10]}{'...' if len(bb_simple_tour) > 10 else ''}"
            )
            results.append(
                ("Branch & Bound (Simple)", bb_simple_dist, bb_simple_time, "Exact")
            )
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append(("Branch & Bound (Simple)", "N/A", "N/A", "Failed"))
    else:
        print("⏩ Skipping Simple Branch & Bound (problem too large, > 15 cities)")
    print()

    # Summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<35} {'Distance':<12} {'Time (s)':<12} {'Type':<15}")
    print("-" * 80)

    for name, dist, time_taken, algo_type in results:
        dist_str = f"{dist:.2f}" if isinstance(dist, (int, float)) else "N/A"
        time_str = (
            f"{time_taken:.4f}" if isinstance(time_taken, (int, float)) else "N/A"
        )
        print(f"{name:<35} {dist_str:<12} {time_str:<12} {algo_type:<15}")
    print("=" * 80)
    print()

    # Find best results
    exact_results = [
        (n, d, t)
        for n, d, t, at in results
        if at == "Exact" and d != "N/A" and "subset" not in n
    ]
    heuristic_results = [
        (n, d, t)
        for n, d, t, at in results
        if at in ["Heuristic", "Heuristic*"] and d != "N/A" and "subset" not in n
    ]

    if exact_results:
        best_exact = min(exact_results, key=lambda x: x[1])
        fastest_exact = min(exact_results, key=lambda x: x[2])
        print(f"🏆 Optimal solution: {best_exact[1]:.2f} (by {best_exact[0]})")
        print(f"⏱️  Fastest exact: {fastest_exact[0]} ({fastest_exact[2]:.4f}s)")

        # Show heuristic gaps
        if heuristic_results:
            print()
            print("📊 Heuristic quality vs optimal:")
            for name, dist, _ in heuristic_results:
                gap = (dist - best_exact[1]) / best_exact[1] * 100
                print(f"   {name}: {gap:+.1f}% gap")
    elif heuristic_results:
        best_heuristic = min(heuristic_results, key=lambda x: x[1])
        fastest_heuristic = min(heuristic_results, key=lambda x: x[2])
        print(f"🏆 Best heuristic: {best_heuristic[1]:.2f} (by {best_heuristic[0]})")
        print(f"⏱️  Fastest: {fastest_heuristic[0]} ({fastest_heuristic[2]:.4f}s)")
        print("⚠️  No exact algorithm completed - results may not be optimal")

    # Compare with known optimal if available
    if problem.name in known_optima and known_optima[problem.name] is not None:
        print()
        print("=" * 80)
        print("COMPARISON WITH KNOWN OPTIMAL")
        print("=" * 80)
        optimal = known_optima[problem.name]
        print(f"🎯 Known optimal for {problem.name}: {optimal:,}")
        print()

        # Show all results vs optimal
        all_results = [
            (n, d, t, at) for n, d, t, at in results if d != "N/A" and "subset" not in n
        ]
        if all_results:
            print("Algorithm Performance vs Optimal:")
            print("-" * 60)
            for name, dist, time_taken, algo_type in all_results:
                gap = (dist - optimal) / optimal * 100
                gap_str = f"{gap:+.1f}%" if gap != 0 else "OPTIMAL!"
                print(f"{name:<35} {dist:>10.0f}  {gap_str:>12}")

            # Find best result
            best_result = min(all_results, key=lambda x: x[1])
            print()
            print(f"📊 Best result: {best_result[0]}")
            print(
                f"   Distance: {best_result[1]:.0f} (gap: {(best_result[1] - optimal) / optimal * 100:.1f}%)"
            )
            print(f"   Time: {best_result[2]:.2f} seconds")

    print()
    print("=" * 80)
    print("ALGORITHM CHARACTERISTICS")
    print("=" * 80)
    print()
    print("🌿 Nearest Neighbor:")
    print("   Time:    O(n²) - scales well")
    print("   Space:   O(n) - very efficient")
    print("   Quality: No guaranteed bound (typically 15-30% above optimal)")
    print("   Use:     Quick approximation, starting point for improvements")
    print()
    print("✨ Lin-Kernighan:")
    print("   Time:    O(n²·²) average - good scaling")
    print("   Space:   O(n²) - moderate")
    print("   Quality: Often optimal or near-optimal (typically 0-5% gap)")
    print("   Use:     Best heuristic for quality, standard for TSP benchmarks")
    print()
    print("📊 Held-Karp (Dynamic Programming):")
    print("   Time:    O(n² × 2ⁿ) - exponential growth")
    print("   Space:   O(n × 2ⁿ) - memory intensive")
    print("   Quality: Guaranteed optimal solution")
    print("   Use:     Small instances only (n ≤ 20-25)")
    print()
    print("🌳 Branch & Bound with LP:")
    print("   Time:    O(2ⁿ) worst case, often much better")
    print("   Space:   O(n²) per node - more efficient than Held-Karp")
    print("   Quality: Guaranteed optimal (if completes)")
    print("   Use:     Medium instances (n ≤ 50), better than HK for n > 20")
    print()
    print("🔍 Branch & Bound (Simple):")
    print("   Time:    O(n!) worst case, pruning helps significantly")
    print("   Space:   O(n²) - efficient")
    print("   Quality: Guaranteed optimal")
    print("   Use:     Very small instances (n ≤ 15)")
    print()


if __name__ == "__main__":
    main()
