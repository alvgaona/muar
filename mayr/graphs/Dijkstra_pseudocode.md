# Dijkstra's Algorithm for Shortest Path Problem

## Overview
Dijkstra's algorithm finds the shortest path from a source vertex to all other vertices in a weighted graph with non-negative edge weights. It uses a greedy approach with a priority queue to efficiently explore the graph.

## Pseudocode

```
ALGORITHM: Dijkstra
INPUT:
    graph G = (V, E, W)  // Graph with vertices V, edges E, and weights W
    start_node s ∈ V     // Source vertex
    end_node t ∈ V       // Optional target vertex (for early termination)
    
OUTPUT:
    distances           // Shortest distances from s to all vertices
    previous           // Previous node in shortest path for path reconstruction

BEGIN Dijkstra
    // ==========================================
    // INITIALIZATION
    // ==========================================
    FOR EACH vertex v ∈ V DO
        distances[v] ← ∞
        previous[v] ← NULL
    END FOR
    
    distances[start_node] ← 0
    
    // Priority queue of (distance, vertex) pairs
    PQ ← MinHeap()
    PQ.insert((0, start_node))
    
    visited ← ∅
    
    // ==========================================
    // MAIN ALGORITHM
    // ==========================================
    WHILE PQ ≠ ∅ DO
        (current_dist, current_node) ← PQ.extract_min()
        
        // Skip if already visited
        IF current_node ∈ visited THEN
            CONTINUE
        END IF
        
        visited ← visited ∪ {current_node}
        
        // Early termination if target reached
        IF end_node ≠ NULL AND current_node = end_node THEN
            BREAK
        END IF
        
        // Skip if we found a better path already
        IF current_dist > distances[current_node] THEN
            CONTINUE
        END IF
        
        // Relaxation step: check all neighbors
        FOR EACH neighbor ∈ Neighbors(current_node) DO
            IF neighbor ∈ visited THEN
                CONTINUE
            END IF
            
            // Calculate tentative distance
            edge_weight ← W(current_node, neighbor)
            new_dist ← current_dist + edge_weight
            
            // Update if shorter path found
            IF new_dist < distances[neighbor] THEN
                distances[neighbor] ← new_dist
                previous[neighbor] ← current_node
                PQ.insert((new_dist, neighbor))
            END IF
        END FOR
    END WHILE
    
    RETURN (distances, previous)
END Dijkstra


// ==========================================
// PATH RECONSTRUCTION SUBROUTINE
// ==========================================
ALGORITHM: ReconstructPath
INPUT:
    previous          // Previous node mapping from Dijkstra
    start_node s      // Source vertex
    end_node t        // Target vertex
    
OUTPUT:
    path             // Shortest path from s to t

BEGIN ReconstructPath
    path ← []
    current ← end_node
    
    // Trace back from end to start
    WHILE current ≠ NULL DO
        path.prepend(current)
        current ← previous[current]
    END WHILE
    
    // Verify path validity
    IF path[0] ≠ start_node THEN
        RETURN []  // No path exists
    END IF
    
    RETURN path
END ReconstructPath
```

## Algorithm Components

### 1. Data Structures
- **Distance Array**: Tracks shortest known distance to each vertex
- **Previous Array**: Stores predecessor for path reconstruction
- **Priority Queue**: Min-heap ordered by distance for efficient vertex selection
- **Visited Set**: Tracks processed vertices to avoid redundant work

### 2. Key Operations
- **Initialization**: Set all distances to infinity except source (0)
- **Relaxation**: Update distance if shorter path found through current vertex
- **Greedy Selection**: Always process vertex with minimum distance next
- **Early Termination**: Optional - stop when target vertex is reached

## Correctness Principle

Dijkstra's algorithm is based on the **greedy choice property**:
- Once a vertex is visited (extracted from priority queue with minimum distance), its shortest path has been found
- This works because:
  1. All edge weights are non-negative
  2. Any path through unvisited vertices would be longer
  3. The algorithm maintains the invariant that distances[v] is the shortest path to v using only visited vertices

## Time Complexity

| Implementation | Time Complexity | Notes |
|----------------|-----------------|-------|
| Binary Heap | O((V + E) log V) | Standard implementation |
| Fibonacci Heap | O(E + V log V) | Theoretically optimal |
| Array | O(V²) | Simple but slower for sparse graphs |

Where:
- V = number of vertices
- E = number of edges

### Detailed Analysis (Binary Heap):
- Initialization: O(V)
- Each vertex extracted once: O(V log V)
- Each edge relaxed once: O(E log V)
- **Total**: O((V + E) log V)

## Space Complexity

- Distance array: O(V)
- Previous array: O(V)
- Priority queue: O(V)
- Visited set: O(V)
- **Total**: O(V)

## Advantages

1. **Optimal solution** for single-source shortest path
2. **Efficient** for sparse graphs with heap implementation
3. **Simple to implement** and understand
4. **Versatile** - can find paths to all vertices or stop at target
5. **Path reconstruction** possible with previous array

## Limitations

1. **Cannot handle negative edge weights** - use Bellman-Ford instead
2. **Single-source only** - for all-pairs use Floyd-Warshall
3. **Requires non-negative weights** - fundamental assumption
4. **Memory intensive** for very large graphs

## Variants and Extensions

### 1. Bidirectional Dijkstra
- Run from both source and target simultaneously
- Stop when searches meet
- Can reduce search space significantly

### 2. A* Algorithm
- Add heuristic function h(v) to guide search
- Priority = g(v) + h(v) where g(v) is distance from start
- Faster for point-to-point queries with good heuristic

### 3. Dial's Algorithm
- For small integer weights
- Uses buckets instead of heap
- O(E + V × W) where W is maximum weight

## Practical Considerations

### 1. Graph Representation
- **Adjacency List**: Better for sparse graphs (E << V²)
- **Adjacency Matrix**: Better for dense graphs (E ≈ V²)

### 2. Priority Queue Implementation
```python
# Binary heap (standard)
import heapq
pq = [(0, start)]
while pq:
    dist, node = heapq.heappop(pq)
    # process node
    heapq.heappush(pq, (new_dist, neighbor))
```

### 3. Early Termination
```
IF only need path to specific target THEN
    BREAK when target is visited
    Saves unnecessary computation
```

### 4. Preprocessing for Multiple Queries
- **Contraction Hierarchies**: Preprocess graph structure
- **Hub Labeling**: Store distance labels
- **Transit Node Routing**: Identify important nodes

## Implementation Tips

### 1. Handle Disconnected Graphs
```
After algorithm completes:
IF distances[target] = ∞ THEN
    No path exists
```

### 2. Avoid Duplicate Entries in Priority Queue
```
Use visited set to skip already processed nodes
OR use decrease-key operation if available
```

### 3. Path Storage Optimization
```
Store only previous pointers, not full paths
Reconstruct path only when needed
```

## Example Applications

### 1. GPS Navigation
- Find shortest route between locations
- Consider traffic as edge weights
- Update dynamically with real-time data

### 2. Network Routing
- OSPF (Open Shortest Path First) protocol
- Find optimal packet routes
- Minimize latency or hop count

### 3. Game Pathfinding
- NPC movement in video games
- Navigate around obstacles
- Often combined with A* for efficiency

### 4. Social Network Analysis
- Find degrees of separation
- Identify shortest connection paths
- Analyze network connectivity

### 5. Flight Planning
- Minimize flight time or cost
- Consider connections and layovers
- Handle time-dependent weights

## Example Execution

For a simple graph with 4 vertices:
```
Graph: A--5--B
       |     |
       3     2
       |     |
       C--1--D

Start: A
1. Initialize: dist[A]=0, dist[B]=∞, dist[C]=∞, dist[D]=∞
2. Process A: Update dist[B]=5, dist[C]=3
3. Process C: Update dist[D]=4
4. Process D: Update dist[B]=5 (no change)
5. Process B: Complete
Result: Shortest paths from A: B=5, C=3, D=4
```

## Performance Comparison

| Graph Type | Best Algorithm | Time Complexity |
|------------|---------------|-----------------|
| Sparse, non-negative | Dijkstra (binary heap) | O((V+E) log V) |
| Dense, non-negative | Dijkstra (array) | O(V²) |
| Negative weights | Bellman-Ford | O(VE) |
| All-pairs | Floyd-Warshall | O(V³) |
| Point-to-point with heuristic | A* | O(E) best case |

## Summary

Dijkstra's algorithm is the gold standard for single-source shortest path problems:
- **Guarantee**: Always finds optimal paths
- **Efficiency**: O((V+E) log V) with binary heap
- **Simplicity**: Straightforward greedy approach
- **Limitation**: Requires non-negative weights
- **Applications**: Ubiquitous in routing and navigation

The algorithm elegantly combines greedy selection with dynamic programming principles to efficiently solve a fundamental graph problem.