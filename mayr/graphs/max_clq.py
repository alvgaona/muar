def read_dimacs_graph(filename: str) -> dict[int, set[int]]:
    """
    Read a graph from a DIMACS format .clq file.

    Args:
        filename: Path to the DIMACS format file

    Returns:
        Dictionary representing adjacency list {node: set(neighbors)}
    """
    graph = {}
    num_nodes = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Comment lines
            if line.startswith("c"):
                continue

            # Problem line: p edge <num_nodes> <num_edges>
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 3:
                    num_nodes = int(parts[2])
                    # Initialize all nodes with empty neighbor sets
                    for i in range(1, num_nodes + 1):
                        graph[i] = set()
                continue

            # Edge line: e <node1> <node2>
            if line.startswith("e"):
                parts = line.split()
                if len(parts) >= 3:
                    node1 = int(parts[1])
                    node2 = int(parts[2])

                    # Ensure nodes exist in graph
                    if node1 not in graph:
                        graph[node1] = set()
                    if node2 not in graph:
                        graph[node2] = set()

                    # Add edge (undirected graph)
                    graph[node1].add(node2)
                    graph[node2].add(node1)

    return graph


def bron_kerbosch_basic(
    graph: dict[int, set[int]],
    r: set[int] | None = None,
    p: set[int] | None = None,
    x: set[int] | None = None,
) -> list[set[int]]:
    """
    Basic Bron-Kerbosch algorithm to find all maximal cliques.
    Simple but can be slow and memory-intensive.

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}
        r: Current clique being built (set)
        p: Candidate nodes that can extend the clique (set)
        x: Nodes already processed (set)

    Returns:
        List of sets, each representing a maximal clique
    """
    if r is None:
        r = set()
    if p is None:
        p = set(graph.keys())
    if x is None:
        x = set()

    cliques: list[set[int]] = []

    # Base case: no more candidates and no exclusions
    if not p and not x:
        cliques.append(r)
        return cliques

    # Iterate over a copy of p since we'll modify it
    for v in list(p):
        neighbors = graph[v]
        # Recursively find cliques and accumulate results
        sub_cliques = bron_kerbosch_basic(graph, r | {v}, p & neighbors, x & neighbors)
        cliques.extend(sub_cliques)

        # Move v from p to x
        p = p - {v}
        x = x | {v}

    return cliques


def bron_kerbosch_pivot(
    graph: dict[int, set[int]],
    r: set[int] | None = None,
    p: set[int] | None = None,
    x: set[int] | None = None,
) -> list[set[int]]:
    """
    Bron-Kerbosch algorithm with pivoting to find all maximal cliques.
    Much faster than basic version due to pivot optimization.

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}
        r: Current clique being built (set)
        p: Candidate nodes that can extend the clique (set)
        x: Nodes already processed (set)

    Returns:
        List of sets, each representing a maximal clique
    """
    if r is None:
        r = set()
    if p is None:
        p = set(graph.keys())
    if x is None:
        x = set()

    cliques: list[set[int]] = []

    # Base case: no more candidates and no exclusions
    if not p and not x:
        cliques.append(r)
        return cliques

    # Choose pivot with most neighbors in P (reduces recursion)
    pivot = max(p | x, key=lambda u: len(p & graph[u])) if p or x else None

    # Only iterate over nodes not connected to pivot
    candidates = p - graph.get(pivot, set()) if pivot else p

    for v in candidates:
        neighbors = graph[v]
        # Recursively find cliques and accumulate results
        sub_cliques = bron_kerbosch_pivot(graph, r | {v}, p & neighbors, x & neighbors)
        cliques.extend(sub_cliques)

        # Move v from p to x
        p = p - {v}
        x = x | {v}

    return cliques


def branch_and_bound_clique(
    graph: dict[int, set[int]],
    current_clique: list[int],
    candidates: list[int],
    best_clique: list[int],
) -> list[int]:
    """
    Branch-and-bound algorithm to find maximum clique.
    Memory efficient - only tracks the best clique, not all cliques.

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}
        current_clique: Current clique being built
        candidates: Candidate nodes that can extend the clique
        best_clique: Best clique found so far

    Returns:
        The maximum clique found
    """
    # Pruning: if we can't beat the best, stop
    if len(current_clique) + len(candidates) <= len(best_clique):
        return best_clique

    # Base case: no more candidates
    if not candidates:
        if len(current_clique) > len(best_clique):
            return current_clique[:]
        return best_clique

    # Try adding each candidate
    for i, v in enumerate(candidates):
        # New clique with v added
        new_clique = current_clique + [v]

        # Find new candidates: nodes connected to all nodes in new_clique
        new_candidates = [
            u
            for u in candidates[i + 1 :]
            if all(u in graph[node] for node in new_clique)
        ]

        # Recurse
        best_clique = branch_and_bound_clique(
            graph, new_clique, new_candidates, best_clique
        )

    return best_clique


def greedy_max_clique(graph: dict[int, set[int]]) -> set[int]:
    """
    Greedy heuristic to quickly find a large clique (not guaranteed to be maximum).

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}

    Returns:
        Set of nodes representing a large clique
    """
    # Start with the node with highest degree
    if not graph:
        return set()

    # Sort nodes by degree (descending)
    nodes_by_degree = sorted(graph.keys(), key=lambda n: len(graph[n]), reverse=True)

    clique = set()

    for node in nodes_by_degree:
        # Check if node is connected to all nodes in current clique
        if all(neighbor in graph[node] for neighbor in clique):
            clique.add(node)

    return clique


def grasp_max_clique(
    graph: dict[int, set[int]], iterations: int = 100, alpha: float = 0.2
) -> set[int]:
    """
    GRASP (Greedy Randomized Adaptive Search Procedure) for maximum clique.
    Combines randomized greedy construction with local search.

    Args:
        graph: Dictionary representing adjacency list
        iterations: Number of GRASP iterations
        alpha: Randomization parameter (0=pure greedy, 1=pure random)

    Returns:
        Best clique found
    """
    import random

    best_clique = set()

    for _ in range(iterations):
        # Construction phase: build a clique greedily with randomization
        clique = set()
        candidates = set(graph.keys())

        while candidates:
            # Calculate degrees in remaining candidate set
            degrees = []
            for node in candidates:
                # Count how many candidates this node is connected to
                degree = sum(1 for neighbor in graph[node] if neighbor in candidates)
                degrees.append((degree, node))

            if not degrees:
                break

            degrees.sort(reverse=True)

            # RCL (Restricted Candidate List): top alpha% of candidates
            rcl_size = max(1, int(len(degrees) * alpha))
            rcl = [node for _, node in degrees[:rcl_size]]

            # Randomly select from RCL
            selected = random.choice(rcl)
            clique.add(selected)

            # Update candidates: only nodes connected to all nodes in clique
            candidates = candidates & graph[selected]

        # Local search phase: try to improve
        clique = local_search(graph, clique)

        if len(clique) > len(best_clique):
            best_clique = clique

    return best_clique


def local_search(graph: dict[int, set[int]], initial_clique: set[int]) -> set[int]:
    """
    Local search to improve a clique by swapping nodes.

    Args:
        graph: Dictionary representing adjacency list
        initial_clique: Starting clique

    Returns:
        Improved clique
    """
    clique = initial_clique.copy()
    improved = True

    while improved:
        improved = False
        clique_list = list(clique)

        # Try 1-1 swaps: remove one node, add one node
        for node_out in clique_list:
            # Find nodes that could replace node_out
            remaining = clique - {node_out}
            candidates = set(graph.keys()) - clique

            for node_in in candidates:
                # Check if node_in is connected to all remaining nodes
                if all(neighbor in graph[node_in] for neighbor in remaining):
                    # Found a valid swap - check if it leads to expansion
                    new_clique = remaining | {node_in}

                    # Try to add more nodes
                    expanded = new_clique.copy()
                    for extra in candidates - {node_in}:
                        if all(neighbor in graph[extra] for neighbor in expanded):
                            expanded.add(extra)

                    if len(expanded) > len(clique):
                        clique = expanded
                        improved = True
                        break

            if improved:
                break

    return clique


def ostergard_max_clique(graph: dict[int, set[int]]) -> set[int]:
    """
    Ostergard's algorithm - efficient branch-and-bound with advanced pruning.
    Uses vertex coloring for better bounds.

    Args:
        graph: Dictionary representing adjacency list

    Returns:
        Maximum clique
    """

    def greedy_coloring(nodes: list[int]) -> dict[int, int]:
        """Assign colors to nodes greedily (for upper bound estimation)."""
        colors = {}
        for node in nodes:
            # Find neighbors' colors
            neighbor_colors = {
                colors[n] for n in graph[node] if n in colors and n in nodes
            }
            # Assign first available color
            color = 0
            while color in neighbor_colors:
                color += 1
            colors[node] = color
        return colors

    def search(
        candidates: list[int], current_clique: list[int], best: list[int]
    ) -> list[int]:
        """Recursive search with pruning."""
        if not candidates:
            return current_clique if len(current_clique) > len(best) else best

        # Color-based pruning: if colors + current size <= best, prune
        colors = greedy_coloring(candidates)
        max_colors = max(colors.values()) + 1 if colors else 0

        if len(current_clique) + max_colors <= len(best):
            return best

        # Try adding each candidate
        for i, node in enumerate(candidates):
            # Pruning by color: skip if can't improve
            node_color = colors.get(node, 0)
            if len(current_clique) + node_color + 1 <= len(best):
                continue

            # New candidates: nodes connected to all in current clique + node
            new_clique = current_clique + [node]
            new_candidates = [
                c for c in candidates[i + 1 :] if all(c in graph[v] for v in new_clique)
            ]

            best = search(new_candidates, new_clique, best)

        return best

    # Sort by degree descending for better initial bound
    candidates = sorted(graph.keys(), key=lambda n: len(graph[n]), reverse=True)

    # Start with greedy solution
    initial = list(greedy_max_clique(graph))

    result = search(candidates, [], initial)
    return set(result)


def find_max_clique(graph: dict[int, set[int]]) -> set[int]:
    """
    Find the maximum clique in the graph using branch-and-bound.
    Memory efficient - doesn't store all cliques.

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}

    Returns:
        Set of nodes representing the maximum clique
    """
    if not graph:
        return set()

    # Sort nodes by degree (descending) for better pruning
    candidates = sorted(graph.keys(), key=lambda n: len(graph[n]), reverse=True)

    # Start with greedy solution as initial bound
    best_clique = list(greedy_max_clique(graph))

    # Run branch-and-bound
    result = branch_and_bound_clique(graph, [], candidates, best_clique)

    return set(result)


if __name__ == "__main__":
    import time

    # You can change this to test different graphs
    filename = "./data/keller5.clq.txt"
    # filename = "./data/p_hat300-2.clq.txt"
    # filename = "./data/brock200_2.clq.txt"
    # filename = "./data/test20.clq.txt"

    print("=" * 80)
    print("MAXIMUM CLIQUE ALGORITHM COMPARISON")
    print("=" * 80)
    print()
    print(f"Reading graph from {filename}...")
    graph = read_dimacs_graph(filename)

    print("Graph loaded:")
    print(f"  Number of nodes: {len(graph)}")
    num_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
    print(f"  Number of edges: {num_edges}")
    print(f"  Edge density: {num_edges / (len(graph) * (len(graph) - 1) / 2):.3f}")
    print()

    results = []

    # Algorithm 1: Greedy Heuristic
    print("=" * 80)
    print("Algorithm 1: GREEDY HEURISTIC")
    print("=" * 80)
    print("Description: Fast approximation - picks nodes greedily by degree")
    print()
    start_time = time.time()
    greedy_clique = greedy_max_clique(graph)
    greedy_time = time.time() - start_time
    print(f"✓ Completed in {greedy_time:.4f} seconds")
    print(f"  Clique size: {len(greedy_clique)}")
    print(
        f"  Nodes: {sorted(greedy_clique)[:10]}{'...' if len(greedy_clique) > 10 else ''}"
    )
    print()
    results.append(("Greedy Heuristic", len(greedy_clique), greedy_time, "Approximate"))

    # Algorithm 2: GRASP
    print("=" * 80)
    print("Algorithm 2: GRASP (Greedy Randomized Adaptive Search)")
    print("=" * 80)
    print("Description: Randomized greedy + local search, multiple iterations")
    print()
    start_time = time.time()
    try:
        grasp_clique = grasp_max_clique(graph, iterations=50, alpha=0.3)
        grasp_time = time.time() - start_time
        print(f"✓ Completed in {grasp_time:.4f} seconds")
        print(f"  Clique size: {len(grasp_clique)}")
        print(
            f"  Nodes: {sorted(grasp_clique)[:10]}{'...' if len(grasp_clique) > 10 else ''}"
        )
        results.append(("GRASP", len(grasp_clique), grasp_time, "Heuristic"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("GRASP", "N/A", "N/A", "Failed"))
    print()

    # Algorithm 3: Ostergard's Algorithm
    print("=" * 80)
    print("Algorithm 3: OSTERGARD'S ALGORITHM")
    print("=" * 80)
    print("Description: Advanced branch-and-bound with vertex coloring bounds")
    print()
    start_time = time.time()
    try:
        ost_clique = ostergard_max_clique(graph)
        ost_time = time.time() - start_time
        print(f"✓ Completed in {ost_time:.4f} seconds")
        print(f"  Clique size: {len(ost_clique)}")
        print(
            f"  Nodes: {sorted(ost_clique)[:10]}{'...' if len(ost_clique) > 10 else ''}"
        )
        results.append(("Ostergard", len(ost_clique), ost_time, "Exact"))
    except Exception as e:
        print(f"✗ Failed: {e}")
        results.append(("Ostergard", "N/A", "N/A", "Failed"))
    print()

    # Only run slower algorithms on smaller graphs
    if len(graph) <= 100:
        # Algorithm 4: Basic Bron-Kerbosch
        print("=" * 80)
        print("Algorithm 4: BRON-KERBOSCH (Basic)")
        print("=" * 80)
        print("Description: Finds ALL maximal cliques, then picks largest")
        print()
        start_time = time.time()
        try:
            all_cliques = bron_kerbosch_basic(graph)
            bk_clique = max(all_cliques, key=len)
            bk_time = time.time() - start_time
            print(f"✓ Completed in {bk_time:.4f} seconds")
            print(f"  Total maximal cliques found: {len(all_cliques)}")
            print(f"  Maximum clique size: {len(bk_clique)}")
            print(
                f"  Nodes: {sorted(bk_clique)[:10]}{'...' if len(bk_clique) > 10 else ''}"
            )
            results.append(("Bron-Kerbosch (Basic)", len(bk_clique), bk_time, "Exact"))
        except Exception as e:
            print(f"✗ Failed or took too long: {e}")
            results.append(("Bron-Kerbosch (Basic)", "N/A", "N/A", "Failed"))
        print()

        # Algorithm 5: Bron-Kerbosch with Pivoting
        print("=" * 80)
        print("Algorithm 5: BRON-KERBOSCH (with Pivoting)")
        print("=" * 80)
        print("Description: Optimized version with pivot selection")
        print()
        start_time = time.time()
        try:
            all_cliques_pivot = bron_kerbosch_pivot(graph)
            bkp_clique = max(all_cliques_pivot, key=len)
            bkp_time = time.time() - start_time
            print(f"✓ Completed in {bkp_time:.4f} seconds")
            print(f"  Total maximal cliques found: {len(all_cliques_pivot)}")
            print(f"  Maximum clique size: {len(bkp_clique)}")
            print(
                f"  Nodes: {sorted(bkp_clique)[:10]}{'...' if len(bkp_clique) > 10 else ''}"
            )
            results.append(
                ("Bron-Kerbosch (Pivot)", len(bkp_clique), bkp_time, "Exact")
            )
        except Exception as e:
            print(f"✗ Failed or took too long: {e}")
            results.append(("Bron-Kerbosch (Pivot)", "N/A", "N/A", "Failed"))
        print()

        # Algorithm 6: Branch-and-Bound
        print("=" * 80)
        print("Algorithm 6: BRANCH-AND-BOUND")
        print("=" * 80)
        print(
            "Description: Memory-efficient, finds only maximum clique (no storage of all)"
        )
        print()
        start_time = time.time()
        try:
            bb_clique = find_max_clique(graph)
            bb_time = time.time() - start_time
            print(f"✓ Completed in {bb_time:.4f} seconds")
            print(f"  Maximum clique size: {len(bb_clique)}")
            print(
                f"  Nodes: {sorted(bb_clique)[:10]}{'...' if len(bb_clique) > 10 else ''}"
            )
            results.append(("Branch-and-Bound", len(bb_clique), bb_time, "Exact"))
        except Exception as e:
            print(f"✗ Failed or took too long: {e}")
            results.append(("Branch-and-Bound", "N/A", "N/A", "Failed"))
        print()
    else:
        print("⏩ Skipping Bron-Kerbosch and basic Branch-and-Bound (graph too large)")
        print()

    # Summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Algorithm':<30} {'Size':<8} {'Time (s)':<12} {'Type':<12}")
    print("-" * 80)
    for name, size, time_taken, algo_type in results:
        size_str = str(size) if size != "N/A" else "N/A"
        time_str = f"{time_taken:.4f}" if time_taken != "N/A" else "N/A"
        print(f"{name:<30} {size_str:<8} {time_str:<12} {algo_type:<12}")
    print("=" * 80)
    print()

    # Find best results
    exact_results = [
        (n, s, t) for n, s, t, at in results if at == "Exact" and t != "N/A"
    ]
    heuristic_results = [
        (n, s, t)
        for n, s, t, at in results
        if at in ["Approximate", "Heuristic"] and t != "N/A"
    ]

    if exact_results:
        fastest = min(exact_results, key=lambda x: x[2])
        print(f"🏆 Fastest exact algorithm: {fastest[0]} ({fastest[2]:.4f}s)")
        print(f"📊 Optimal clique size: {fastest[1]}")
    elif heuristic_results:
        best_heuristic = max(heuristic_results, key=lambda x: x[1])
        print(
            f"🏆 Best heuristic result: {best_heuristic[0]} (size {best_heuristic[1]} in {best_heuristic[2]:.4f}s)"
        )
        print("⚠️  No exact algorithms completed - result may not be optimal")
    print()
