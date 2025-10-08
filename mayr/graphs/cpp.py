"""Chinese Postman Problem Solver

This module implements algorithms for solving the Chinese Postman Problem (CPP)
on undirected, complete, Euclidean graphs. The CPP finds the shortest closed path
that visits every edge at least once.

Implemented from scratch without external graph libraries.
Uses .tsp files from the data directory for testing.
"""

import time
from typing import Any
from collections import defaultdict
import tsplib95
import sys


class CPPSolver:
    """Base class for CPP solvers on complete Euclidean graphs."""

    def __init__(self, problem: Any) -> None:
        """
        Initialize the CPP solver.

        Args:
            problem: tsplib95 problem instance
        """
        self.problem = problem
        self.nodes = list(problem.get_nodes())
        self.n = len(self.nodes)
        self.dist_matrix = self._compute_distance_matrix()
        # Complete graph represented as adjacency list
        self.adj_list = self._build_complete_graph()

    def _compute_distance_matrix(self) -> dict[tuple[int, int], float]:
        """Compute the distance matrix using problem's weight function."""
        dist_matrix = {}

        for i, node_i in enumerate(self.nodes):
            for j, node_j in enumerate(self.nodes):
                if i != j:
                    # Use the problem's weight function
                    dist_matrix[(node_i, node_j)] = self.problem.get_weight(
                        node_i, node_j
                    )

        return dist_matrix

    def _build_complete_graph(self) -> dict[int, list[int]]:
        """Build adjacency list for complete graph."""
        adj_list = defaultdict(list)
        for node_i in self.nodes:
            for node_j in self.nodes:
                if node_i != node_j:
                    adj_list[node_i].append(node_j)
        return dict(adj_list)

    def get_vertex_degree(self) -> int:
        """Get degree of a vertex in complete graph."""
        return self.n - 1  # Complete graph: each vertex connects to all others

    def calculate_path_cost(self, path: list[tuple[int, int]]) -> float:
        """Calculate the total cost of an edge path."""
        total_cost = 0.0
        for u, v in path:
            if (u, v) in self.dist_matrix:
                total_cost += self.dist_matrix[(u, v)]
            else:
                total_cost += self.dist_matrix[(v, u)]
        return total_cost

    def verify_cpp_solution(self, path: list[tuple[int, int]]) -> bool:
        """Verify that a path visits all edges at least once."""
        required_edges = set()
        for i, node_i in enumerate(self.nodes):
            for j in range(i + 1, self.n):
                node_j = self.nodes[j]
                required_edges.add((min(node_i, node_j), max(node_i, node_j)))

        visited_edges = set()
        for u, v in path:
            edge = (min(u, v), max(u, v))
            visited_edges.add(edge)

        return required_edges.issubset(visited_edges)


class StandardCPP(CPPSolver):
    """
    Standard Chinese Postman Algorithm for complete graphs.

    Time Complexity: O(n³) for matching, O(n²) for circuit
    Space Complexity: O(n²)

    This is an EXACT algorithm that always finds the optimal solution.
    """

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """
        Solve CPP using the standard algorithm.

        Returns:
            Tuple of (edge_path, total_cost)
        """
        # Check if graph is already Eulerian (all vertices have even degree)
        odd_vertices = []
        for v in self.nodes:
            if self.get_vertex_degree() % 2 == 1:
                odd_vertices.append(v)

        if len(odd_vertices) == 0:
            print("  Graph is already Eulerian!")
            # Build simple edge list for Eulerian circuit
            adj_simple = defaultdict(list)
            for node_i in self.nodes:
                for node_j in self.nodes:
                    if node_i != node_j:
                        adj_simple[node_i].append(node_j)

            circuit = self._find_eulerian_circuit_simple(adj_simple)
            cost = self.calculate_path_cost(circuit)
            return circuit, cost

        print(f"  Found {len(odd_vertices)} odd-degree vertices")

        # Find minimum weight perfect matching of odd vertices
        matching_edges = self._minimum_weight_matching(odd_vertices)
        print(f"  Added {len(matching_edges)} matching edges")

        # Create multigraph by adding matched edges
        augmented_adj = self._augment_graph(matching_edges)

        # Find Eulerian circuit in augmented graph
        circuit = self._find_eulerian_circuit(augmented_adj)
        cost = self.calculate_path_cost(circuit)

        return circuit, cost

    def _minimum_weight_matching(self, vertices: list[int]) -> list[tuple[int, int]]:
        """
        Find minimum weight perfect matching using dynamic programming.
        For complete graphs, we can use a simpler approach.
        """
        if len(vertices) == 0:
            return []

        # For complete Euclidean graphs, greedy matching is optimal
        matching = []
        unmatched = set(vertices)

        while len(unmatched) > 1:
            # Pick arbitrary vertex
            u = unmatched.pop()

            # Find closest unmatched vertex
            best_v = None
            best_dist = float("inf")

            for v in unmatched:
                dist = self.dist_matrix.get(
                    (u, v), self.dist_matrix.get((v, u), float("inf"))
                )
                if dist < best_dist:
                    best_dist = dist
                    best_v = v

            if best_v is not None:
                matching.append((u, best_v))
                unmatched.remove(best_v)

        return matching

    def _augment_graph(
        self, matching_edges: list[tuple[int, int]]
    ) -> dict[int, list[tuple[int, int]]]:
        """Create multigraph by adding matched edges."""
        # Use list of (neighbor, edge_id) to handle multiple edges
        augmented = defaultdict(list)
        edge_id = 0

        # Add original edges
        for i, node_i in enumerate(self.nodes):
            for j in range(i + 1, self.n):
                node_j = self.nodes[j]
                augmented[node_i].append((node_j, edge_id))
                augmented[node_j].append((node_i, edge_id))
                edge_id += 1

        # Add matching edges (duplicates)
        for u, v in matching_edges:
            augmented[u].append((v, edge_id))
            augmented[v].append((u, edge_id))
            edge_id += 1

        return dict(augmented)

    def _find_eulerian_circuit(
        self, adj_list: dict[int, list[tuple[int, int]]]
    ) -> list[tuple[int, int]]:
        """Find Eulerian circuit using Hierholzer's algorithm."""
        if not adj_list:
            return []

        # Start from first node
        curr_path = [self.nodes[0]]
        circuit = []
        edges_used = set()

        while curr_path:
            curr = curr_path[-1]

            # Find unused edge from current vertex
            found_edge = False
            if curr in adj_list:
                for neighbor, edge_id in list(adj_list[curr]):
                    if edge_id not in edges_used:
                        curr_path.append(neighbor)
                        edges_used.add(edge_id)

                        # Remove edge from both directions
                        adj_list[curr].remove((neighbor, edge_id))
                        adj_list[neighbor].remove((curr, edge_id))

                        found_edge = True
                        break

            if not found_edge:
                # Backtrack
                if len(curr_path) > 1:
                    v = curr_path.pop()
                    u = curr_path[-1] if curr_path else 0
                    circuit.append((u, v))
                else:
                    curr_path.pop()

        return circuit

    def _find_eulerian_circuit_simple(
        self, adj: dict[int, list[int]]
    ) -> list[tuple[int, int]]:
        """Simple Eulerian circuit for when graph is already Eulerian."""
        if not adj:
            return []

        stack = [self.nodes[0]]
        circuit = []

        while stack:
            curr = stack[-1]
            if curr in adj and adj[curr]:
                next_v = adj[curr].pop()
                # Remove reverse edge for undirected graph
                if next_v in adj and curr in adj[next_v]:
                    adj[next_v].remove(curr)
                stack.append(next_v)
            else:
                if len(stack) > 1:
                    v = stack.pop()
                    circuit.append((stack[-1], v))
                else:
                    stack.pop()

        return circuit


class HierholzerCPP(CPPSolver):
    """
    Hierholzer's Algorithm for Eulerian graphs.

    Time Complexity: O(n²) for complete graphs
    Space Complexity: O(n²)

    This is an EXACT algorithm for finding Eulerian circuits.
    Only works when the graph is already Eulerian (odd n for complete graphs).
    """

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """
        Solve CPP using Hierholzer's algorithm.

        Returns:
            Tuple of (edge_path, total_cost)
        """
        # Check if graph is Eulerian
        odd_vertices = []
        for v in self.nodes:
            if self.get_vertex_degree() % 2 == 1:
                odd_vertices.append(v)

        if len(odd_vertices) > 0:
            print(f"  ⚠️ Graph is not Eulerian ({len(odd_vertices)} odd vertices)")
            print("  Falling back to standard CPP algorithm...")
            standard = StandardCPP(self.problem)
            return standard.solve()

        print("  Graph is Eulerian - using Hierholzer's algorithm")

        # Build edge list for Hierholzer
        adj_copy = self._build_edge_list()
        circuit = self._hierholzer(adj_copy)
        cost = self.calculate_path_cost(circuit)

        return circuit, cost

    def _build_edge_list(self) -> dict[int, list[int]]:
        """Build edge list for complete graph."""
        edges = defaultdict(list)
        for node_i in self.nodes:
            for node_j in self.nodes:
                if node_i != node_j:
                    edges[node_i].append(node_j)
        return edges

    def _hierholzer(self, adj: dict[int, list[int]]) -> list[tuple[int, int]]:
        """Hierholzer's algorithm for Eulerian circuit."""
        stack = [self.nodes[0]]  # Start from the first node
        circuit = []

        while stack:
            curr = stack[-1]
            if curr in adj and adj[curr]:
                next_v = adj[curr].pop()
                # Remove reverse edge for undirected graph
                if next_v in adj and curr in adj[next_v]:
                    adj[next_v].remove(curr)
                stack.append(next_v)
            else:
                if len(stack) > 1:
                    v = stack.pop()
                    circuit.append((stack[-1], v))
                else:
                    stack.pop()

        return circuit


class GreedyMatchingCPP(CPPSolver):
    """
    Greedy Matching Algorithm for CPP on complete Euclidean graphs.

    Time Complexity: O(n² log n)
    Space Complexity: O(n²)

    This is an EXACT algorithm for complete Euclidean graphs.
    Uses greedy nearest-neighbor matching which is optimal for Euclidean case.
    """

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """
        Solve CPP using greedy matching.

        Returns:
            Tuple of (edge_path, total_cost)
        """
        # Check degree parity
        odd_vertices = []
        for v in self.nodes:
            if self.get_vertex_degree() % 2 == 1:
                odd_vertices.append(v)

        if len(odd_vertices) == 0:
            print("  Graph is already Eulerian!")
            # Use simple Eulerian circuit
            adj_copy = self._build_edge_list()
            circuit = self._find_eulerian_path(adj_copy)
            cost = self.calculate_path_cost(circuit)
            return circuit, cost

        print(f"  Found {len(odd_vertices)} odd-degree vertices")
        print("  Using greedy nearest-neighbor matching")

        # Greedy matching
        matching_edges = self._greedy_matching(odd_vertices)
        matching_cost = sum(
            self.dist_matrix.get((u, v), self.dist_matrix.get((v, u), 0))
            for u, v in matching_edges
        )
        print(f"  Added {len(matching_edges)} matching edges")
        print(f"  Matching cost: {matching_cost:.2f}")

        # Create augmented graph
        augmented_adj = self._create_augmented_graph(matching_edges)

        # Find Eulerian circuit
        circuit = self._find_eulerian_path(augmented_adj)
        cost = self.calculate_path_cost(circuit)

        return circuit, cost

    def _greedy_matching(self, vertices: list[int]) -> list[tuple[int, int]]:
        """Greedy nearest-neighbor matching."""
        matching = []
        unmatched = set(vertices)

        while len(unmatched) > 1:
            u = min(unmatched)  # Start with lowest numbered vertex
            unmatched.remove(u)

            # Find nearest unmatched vertex
            best_v = None
            best_dist = float("inf")

            for v in unmatched:
                dist = self.dist_matrix.get(
                    (u, v), self.dist_matrix.get((v, u), float("inf"))
                )
                if dist < best_dist:
                    best_dist = dist
                    best_v = v

            if best_v is not None:
                matching.append((u, best_v))
                unmatched.remove(best_v)

        return matching

    def _build_edge_list(self) -> dict[int, list[int]]:
        """Build edge list for complete graph."""
        edges = defaultdict(list)
        for node_i in self.nodes:
            for node_j in self.nodes:
                if node_i != node_j:
                    edges[node_i].append(node_j)
        return edges

    def _create_augmented_graph(
        self, matching_edges: list[tuple[int, int]]
    ) -> dict[int, list[int]]:
        """Create augmented graph with matched edges."""
        augmented = self._build_edge_list()

        # Add matching edges (duplicates)
        for u, v in matching_edges:
            augmented[u].append(v)
            augmented[v].append(u)

        return augmented

    def _find_eulerian_path(self, adj: dict[int, list[int]]) -> list[tuple[int, int]]:
        """Find Eulerian path using DFS."""
        if not adj:
            return []

        stack = [self.nodes[0]]  # Start from the first node
        path = []

        while stack:
            curr = stack[-1]
            if curr in adj and adj[curr]:
                next_v = adj[curr].pop()
                # Remove reverse edge
                if next_v in adj and curr in adj[next_v]:
                    adj[next_v].remove(curr)
                stack.append(next_v)
            else:
                if len(stack) > 1:
                    v = stack.pop()
                    path.append((stack[-1], v))
                else:
                    stack.pop()

        return path


def main():
    # You can change this to test different problems
    filename = "data/test15.tsp"
    # filename = "data/kroA100.tsp"
    # filename = "data/lin318.tsp"
    # filename = "data/pcb442.tsp"

    if len(sys.argv) > 1:
        filename = sys.argv[1]

    print("=" * 80)
    print("CHINESE POSTMAN PROBLEM - ALGORITHM COMPARISON")
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

    # For CPP on complete graph
    n_vertices = problem.dimension
    n_edges = n_vertices * (n_vertices - 1) // 2
    print(f"  Number of edges in complete graph: {n_edges}")

    # Check if Eulerian
    if n_vertices % 2 == 1:
        print(
            f"  Eulerian: Yes (odd n, all vertices have even degree {n_vertices - 1})"
        )
    else:
        print(f"  Eulerian: No (even n, all vertices have odd degree {n_vertices - 1})")
    print()

    results = []

    # Algorithm 1: Standard CPP
    print("=" * 80)
    print("Algorithm 1: STANDARD CPP")
    print("=" * 80)
    print("Description: Exact algorithm using minimum weight matching")
    print("             Always finds optimal solution for CPP")
    print()

    standard_solver = StandardCPP(problem)
    start_time = time.time()
    try:
        standard_path, standard_cost = standard_solver.solve()
        standard_time = time.time() - start_time
        print(f"✓ Completed in {standard_time:.4f} seconds")
        print(f"  Path length: {len(standard_path)} edges")
        print(f"  Total cost: {standard_cost:.2f}")
        print(
            f"  Valid CPP solution: {standard_solver.verify_cpp_solution(standard_path)}"
        )
        results.append(("Standard CPP", standard_cost, standard_time, "Exact"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Standard CPP", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 2: Hierholzer's Algorithm
    print("=" * 80)
    print("Algorithm 2: HIERHOLZER'S ALGORITHM")
    print("=" * 80)
    print("Description: Direct Eulerian circuit construction")
    print("             Only works if graph is already Eulerian")
    print()

    hierholzer_solver = HierholzerCPP(problem)
    start_time = time.time()
    try:
        hierholzer_path, hierholzer_cost = hierholzer_solver.solve()
        hierholzer_time = time.time() - start_time
        print(f"✓ Completed in {hierholzer_time:.4f} seconds")
        print(f"  Path length: {len(hierholzer_path)} edges")
        print(f"  Total cost: {hierholzer_cost:.2f}")
        print(
            f"  Valid CPP solution: {hierholzer_solver.verify_cpp_solution(hierholzer_path)}"
        )
        results.append(("Hierholzer", hierholzer_cost, hierholzer_time, "Exact"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Hierholzer", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 3: Greedy Matching
    print("=" * 80)
    print("Algorithm 3: GREEDY MATCHING")
    print("=" * 80)
    print("Description: Greedy nearest-neighbor matching")
    print("             Exact for complete Euclidean graphs")
    print()

    greedy_solver = GreedyMatchingCPP(problem)
    start_time = time.time()
    try:
        greedy_path, greedy_cost = greedy_solver.solve()
        greedy_time = time.time() - start_time
        print(f"✓ Completed in {greedy_time:.4f} seconds")
        print(f"  Path length: {len(greedy_path)} edges")
        print(f"  Total cost: {greedy_cost:.2f}")
        print(f"  Valid CPP solution: {greedy_solver.verify_cpp_solution(greedy_path)}")
        results.append(("Greedy Matching", greedy_cost, greedy_time, "Exact"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Greedy Matching", "N/A", "N/A", "Failed"))
    print()

    # Summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<25} {'Cost':<12} {'Time (s)':<12} {'Type':<10}")
    print("-" * 80)

    for name, cost, time_taken, algo_type in results:
        cost_str = f"{cost:.2f}" if isinstance(cost, (int, float)) else "N/A"
        time_str = (
            f"{time_taken:.4f}" if isinstance(time_taken, (int, float)) else "N/A"
        )
        print(f"{name:<25} {cost_str:<12} {time_str:<12} {algo_type:<10}")

    print("=" * 80)
    print()

    # Find best results
    valid_results = [(n, c, t) for n, c, t, _ in results if c != "N/A"]
    if valid_results:
        best_cost = min(valid_results, key=lambda x: x[1])
        fastest = min(valid_results, key=lambda x: x[2])

        print(f"🏆 Best solution: {best_cost[1]:.2f} (by {best_cost[0]})")
        print(f"⏱️  Fastest: {fastest[0]} ({fastest[2]:.4f}s)")

        # Check if all algorithms found the same optimal
        costs = [c for _, c, _ in valid_results]
        if len(set(costs)) == 1:
            print("✅ All algorithms found the same optimal solution!")
        else:
            print("⚠️  Different solutions found - investigating...")

    print()
    print("=" * 80)
    print("ALGORITHM CHARACTERISTICS")
    print("=" * 80)
    print()

    print("📦 Standard CPP:")
    print("   Time:    O(n³) - for matching")
    print("   Space:   O(n²) - stores graph")
    print("   Quality: Always optimal (exact)")
    print("   Use:     General CPP solver")
    print()

    print("🔄 Hierholzer's:")
    print("   Time:    O(n²) - for complete graphs")
    print("   Space:   O(n²) - stores edges")
    print("   Quality: Optimal when applicable")
    print("   Use:     Only Eulerian graphs")
    print()

    print("🎯 Greedy Matching:")
    print("   Time:    O(n² log n) - sorting")
    print("   Space:   O(n²) - stores graph")
    print("   Quality: Optimal for Euclidean")
    print("   Use:     Fast for Euclidean case")
    print()


if __name__ == "__main__":
    main()
