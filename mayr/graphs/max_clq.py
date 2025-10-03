def bron_kerbosch(
    graph: dict[str, set[str]],
    r: set[str] | None=None,
    p: set[str] | None=None,
    x: set[str] | None=None,
) -> list[set[str]]:
    """
    Bron-Kerbosch algorithm to find all maximal cliques.

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

    cliques: list[set[str]] = []

    # Base case: no more candidates and no exclusions
    if not p and not x:
        cliques.append(r)
        return cliques

    # Iterate over a copy of p since we'll modify it
    for v in list(p):
        neighbors = graph[v]
        # Recursively find cliques and accumulate results
        sub_cliques = bron_kerbosch(graph, r | {v}, p & neighbors, x & neighbors)
        cliques.extend(sub_cliques)

        # Move v from p to x
        p = p - {v}
        x = x | {v}

    return cliques


def find_max_clique(graph: dict[str, set[str]]) -> set[str]:
    """
    Find the maximum clique in the graph.

    Args:
        graph: Dictionary representing adjacency list {node: set(neighbors)}

    Returns:
        Set of nodes representing the maximum clique
    """
    max_clique: set[str] = set()

    for clique in bron_kerbosch(graph):
        if len(clique) > len(max_clique):
            max_clique = clique

    return max_clique


if __name__ == "__main__":
    graph = {
        'A': {'B', 'C', 'D'},
        'B': {'A', 'C', 'E'},
        'C': {'A', 'B', 'D', 'E'},
        'D': {'A', 'C'},
        'E': {'B', 'C'}
    }

    print("Graph:", graph)
    print("\nAll maximal cliques:")
    for clique in bron_kerbosch(graph):
        print(f"  {clique}")

    max_clique = find_max_clique(graph)
    print(f"\nMaximum clique: {max_clique}")
    print(f"Size: {len(max_clique)}")
