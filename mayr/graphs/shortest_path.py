import tsplib95
import heapq
from typing import Dict, List, Tuple, Optional


def dijkstra(
    graph, start_node: int, end_node: Optional[int] = None
) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
    """
    Implements Dijkstra's shortest path algorithm.

    Args:
        graph: A tsplib95 problem object with node and edge data
        start_node: The starting node for the shortest path
        end_node: Optional ending node. If provided, algorithm can stop early.

    Returns:
        A tuple of (distances, previous) where:
        - distances: Dict mapping each node to its shortest distance from start
        - previous: Dict mapping each node to the previous node in shortest path
    """
    # Initialize distances and previous nodes
    distances = {node: float("infinity") for node in graph.get_nodes()}
    previous = {node: None for node in graph.get_nodes()}
    distances[start_node] = 0

    # Priority queue: (distance, node)
    pq = [(0, start_node)]
    visited = set()

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        # Skip if already visited
        if current_node in visited:
            continue

        visited.add(current_node)

        # Early termination if we reached the end node
        if end_node is not None and current_node == end_node:
            break

        # Skip if we found a better path already
        if current_dist > distances[current_node]:
            continue

        # Check all neighbors
        for neighbor in graph.get_nodes():
            if neighbor == current_node or neighbor in visited:
                continue

            # Get edge weight
            try:
                weight = graph.get_weight(current_node, neighbor)
            except:
                continue

            # Calculate new distance
            new_dist = current_dist + weight

            # Update if we found a shorter path
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, previous


def reconstruct_path(
    previous: Dict[int, Optional[int]], start_node: int, end_node: int
) -> List[int]:
    """
    Reconstructs the shortest path from start to end using the previous dict.

    Args:
        previous: Dict mapping each node to its previous node in the shortest path
        start_node: The starting node
        end_node: The ending node

    Returns:
        List of nodes representing the path from start to end
    """
    path = []
    current = end_node

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    # Check if path is valid (starts with start_node)
    if path[0] != start_node:
        return []

    return path


def main():
    # Load the TSP file
    problem = tsplib95.load("./data/pcb442.tsp")

    print(f"Problem: {problem.name}")
    print(f"Type: {problem.type}")
    print(f"Dimension: {problem.dimension}")
    print(f"Edge Weight Type: {problem.edge_weight_type}")
    print()

    # Run Dijkstra's algorithm from node 1 to node 50
    start_node = 1
    end_node = 50

    print(f"Running Dijkstra's algorithm from node {start_node} to node {end_node}...")
    distances, previous = dijkstra(problem, start_node, end_node)

    # Reconstruct and display the path
    path = reconstruct_path(previous, start_node, end_node)

    print(f"\nShortest path from {start_node} to {end_node}:")
    print(f"Path: {' -> '.join(map(str, path))}")
    print(f"Total distance: {distances[end_node]:.2f}")
    print()

    # Show distances to first 10 nodes for reference
    print(f"Distances from node {start_node} to first 10 nodes:")
    for node in sorted(list(problem.get_nodes())[:10]):
        if distances[node] != float("infinity"):
            print(f"  Node {node}: {distances[node]:.2f}")


if __name__ == "__main__":
    main()
