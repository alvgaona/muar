# Chinese Postman Problem (CPP) Algorithm

## Overview
The Chinese Postman Problem finds the shortest closed path that visits every edge of a graph at least once. For undirected graphs, the solution involves finding odd-degree vertices, matching them optimally, and then finding an Eulerian circuit in the augmented graph.

## Pseudocode

```
ALGORITHM: StandardCPP
INPUT:
    graph G = (V, E, W)  // Undirected weighted graph
    
OUTPUT:
    circuit              // Sequence of edges forming optimal CPP tour
    total_cost           // Total cost of the tour

BEGIN StandardCPP
    // ==========================================
    // STEP 1: CHECK EULERIAN PROPERTY
    // ==========================================
    odd_vertices ← ∅
    
    FOR EACH vertex v ∈ V DO
        IF degree(v) mod 2 = 1 THEN
            odd_vertices ← odd_vertices ∪ {v}
        END IF
    END FOR
    
    IF |odd_vertices| = 0 THEN
        // Graph is already Eulerian
        circuit ← FindEulerianCircuit(G)
        total_cost ← CalculatePathCost(circuit)
        RETURN (circuit, total_cost)
    END IF
    
    // ==========================================
    // STEP 2: MINIMUM WEIGHT PERFECT MATCHING
    // ==========================================
    // Find shortest paths between all pairs of odd vertices
    shortest_paths ← {}
    FOR EACH u ∈ odd_vertices DO
        FOR EACH v ∈ odd_vertices, v ≠ u DO
            shortest_paths[(u,v)] ← ShortestPath(G, u, v)
        END FOR
    END FOR
    
    // Find minimum weight perfect matching
    matching ← MinimumWeightMatching(odd_vertices, shortest_paths)
    
    // ==========================================
    // STEP 3: AUGMENT GRAPH
    // ==========================================
    augmented_graph ← G
    
    FOR EACH (u, v) ∈ matching DO
        // Add duplicate edges along shortest path from u to v
        path ← shortest_paths[(u,v)]
        FOR EACH edge e ∈ path DO
            augmented_graph.add_edge(e)  // Duplicate edge
        END FOR
    END FOR
    
    // ==========================================
    // STEP 4: FIND EULERIAN CIRCUIT
    // ==========================================
    circuit ← FindEulerianCircuit(augmented_graph)
    total_cost ← CalculatePathCost(circuit)
    
    RETURN (circuit, total_cost)
END StandardCPP


// ==========================================
// MINIMUM WEIGHT MATCHING SUBROUTINE
// ==========================================
ALGORITHM: MinimumWeightMatching
INPUT:
    vertices            // Set of vertices to match (even cardinality)
    distances           // Pairwise distances/costs
    
OUTPUT:
    matching           // Set of edges forming perfect matching

BEGIN MinimumWeightMatching
    IF |vertices| = 0 THEN
        RETURN ∅
    END IF
    
    // For general case: Use Hungarian algorithm or Blossom algorithm
    // For Euclidean complete graphs: Greedy works well
    
    matching ← ∅
    unmatched ← vertices
    
    WHILE |unmatched| > 1 DO
        // Pick arbitrary unmatched vertex
        u ← unmatched.pop()
        
        // Find closest unmatched vertex
        best_v ← NULL
        min_distance ← ∞
        
        FOR EACH v ∈ unmatched DO
            IF distances[(u,v)] < min_distance THEN
                min_distance ← distances[(u,v)]
                best_v ← v
            END IF
        END FOR
        
        IF best_v ≠ NULL THEN
            matching ← matching ∪ {(u, best_v)}
            unmatched ← unmatched \ {best_v}
        END IF
    END WHILE
    
    RETURN matching
END MinimumWeightMatching


// ==========================================
// HIERHOLZER'S ALGORITHM FOR EULERIAN CIRCUIT
// ==========================================
ALGORITHM: FindEulerianCircuit
INPUT:
    graph G = (V, E)    // Graph with all vertices of even degree
    
OUTPUT:
    circuit             // Sequence of edges forming Eulerian circuit

BEGIN FindEulerianCircuit
    // Choose starting vertex
    start_vertex ← arbitrary vertex from V
    
    stack ← [start_vertex]
    circuit ← []
    
    // Create mutable copy of edges
    edges_remaining ← E
    
    WHILE stack ≠ ∅ DO
        current ← stack.top()
        
        IF ∃ edge (current, neighbor) ∈ edges_remaining THEN
            // Follow an unused edge
            stack.push(neighbor)
            edges_remaining ← edges_remaining \ {(current, neighbor)}
            
            // For undirected graph, remove reverse edge too
            edges_remaining ← edges_remaining \ {(neighbor, current)}
        ELSE
            // Backtrack - add edge to circuit
            IF |stack| > 1 THEN
                v ← stack.pop()
                u ← stack.top()
                circuit.append((u, v))
            ELSE
                stack.pop()
            END IF
        END IF
    END WHILE
    
    RETURN circuit
END FindEulerianCircuit


// ==========================================
// PATH COST CALCULATION
// ==========================================
ALGORITHM: CalculatePathCost
INPUT:
    path = [(u₁,v₁), (u₂,v₂), ..., (uₙ,vₙ)]  // Sequence of edges
    W                                          // Weight function
    
OUTPUT:
    total_cost                                 // Sum of edge weights

BEGIN CalculatePathCost
    total_cost ← 0
    
    FOR EACH edge (u, v) ∈ path DO
        total_cost ← total_cost + W(u, v)
    END FOR
    
    RETURN total_cost
END CalculatePathCost
```

## Algorithm Components

### 1. Eulerian Property Check
- **Purpose**: Determine if graph needs augmentation
- **Condition**: Graph is Eulerian ⟺ all vertices have even degree
- **Theorem**: For undirected graphs, either all vertices have even degree, or exactly an even number of vertices have odd degree

### 2. Minimum Weight Perfect Matching
- **Input**: Set of odd-degree vertices (always even count)
- **Goal**: Pair vertices to minimize total matching weight
- **Methods**:
  - Hungarian Algorithm: O(n³)
  - Blossom Algorithm: O(n³)
  - Greedy (for Euclidean): O(n² log n)

### 3. Graph Augmentation
- **Process**: Add duplicate edges along shortest paths between matched vertices
- **Result**: All vertices now have even degree
- **Property**: Augmented graph is Eulerian

### 4. Eulerian Circuit Construction
- **Algorithm**: Hierholzer's algorithm
- **Approach**: DFS with backtracking
- **Guarantee**: Always finds circuit in Eulerian graph

## Time Complexity

| Component | Complexity | Notes |
|-----------|------------|-------|
| Degree Check | O(V + E) | Linear scan |
| All-Pairs Shortest Paths | O(V³) | Floyd-Warshall |
| Minimum Matching | O(V³) | Hungarian/Blossom |
| Graph Augmentation | O(E) | Add edges |
| Eulerian Circuit | O(E) | Hierholzer's |
| **Total** | **O(V³)** | Dominated by matching |

### Special Cases:
- **Complete graphs**: O(V²) edges, so O(V²) for circuit
- **Sparse graphs**: Can use Dijkstra for paths: O(V² log V)
- **Euclidean graphs**: Greedy matching often optimal

## Space Complexity

- Distance matrix: O(V²)
- Augmented graph: O(E) potentially doubled
- Stack for Hierholzer: O(E)
- **Total**: O(V² + E)

## Correctness Principle

The algorithm is correct because:

1. **Eulerian Circuit Exists** ⟺ All vertices have even degree
2. **Odd Vertices Count**: Always even (handshaking lemma)
3. **Perfect Matching**: Always exists for even number of vertices
4. **Augmentation**: Preserves connectivity, makes graph Eulerian
5. **Optimality**: Minimum weight matching minimizes added cost

## Special Cases and Optimizations

### 1. Already Eulerian Graphs
```
IF all vertices have even degree THEN
    Skip matching step
    Directly find Eulerian circuit
```

### 2. Complete Graphs
```
For complete graph Kₙ:
    IF n is odd THEN
        Graph is Eulerian (all degrees = n-1 = even)
    ELSE
        All vertices are odd (need matching)
```

### 3. Euclidean Graphs
```
For Euclidean distances:
    Greedy nearest-neighbor matching is often optimal
    Triangle inequality helps bound solution quality
```

## Variants and Extensions

### 1. Directed CPP
```
Check strong connectivity
Balance in-degree and out-degree
Use min-cost flow for augmentation
```

### 2. Mixed CPP
```
Handle both directed and undirected edges
More complex - NP-hard in general
```

### 3. Windy Postman Problem
```
Different costs for traversing edge in each direction
Asymmetric version of CPP
```

### 4. Rural Postman Problem
```
Only subset of edges must be traversed
NP-hard even for undirected graphs
```

## Practical Considerations

### 1. Preprocessing
- **Component Check**: Ensure graph is connected
- **Self-loops**: Remove or handle separately
- **Multiple edges**: Combine into single weighted edge

### 2. Matching Algorithm Choice
```python
IF |odd_vertices| < 20 THEN
    Use exact algorithm (Hungarian/Blossom)
ELSE IF graph is Euclidean THEN
    Use greedy matching
ELSE
    Use approximation algorithm
```

### 3. Memory Optimization
- For large sparse graphs, use adjacency lists
- Store only upper triangle of distance matrix
- Use iterative Hierholzer to avoid stack overflow

### 4. Numerical Stability
- Use integer weights when possible
- Scale floating-point weights to avoid precision issues
- Check for near-zero differences in matching

## Implementation Tips

### 1. Efficient Odd Vertex Detection
```
odd_vertices = []
for v in vertices:
    if adjacency_list[v].size() % 2 == 1:
        odd_vertices.append(v)
```

### 2. Hierholzer's with Edge Tracking
```
Use edge IDs to handle multiple edges:
edges = {edge_id: (u, v, weight)}
used_edges = set()
```

### 3. Path Reconstruction
```
Store parent pointers during shortest path computation
Reconstruct by following parents from destination to source
```

## Applications

1. **Street Sweeping**: Cover all streets minimally
2. **Snow Plowing**: Clear all roads efficiently
3. **Mail Delivery**: Traverse all routes with minimum distance
4. **Network Testing**: Visit all network links for inspection
5. **PCB Testing**: Test all connections on circuit board
6. **Garbage Collection**: Service all streets optimally

## Quality Measures

### 1. Augmentation Ratio
```
ratio = (total_cost - original_edges_cost) / original_edges_cost
```
- Measures how much extra traversal is needed
- For Eulerian graphs: ratio = 0

### 2. Circuit Efficiency
```
efficiency = original_edges_cost / total_cost
```
- Measures what fraction of tour is mandatory
- Best possible: efficiency = 1.0

### 3. Matching Quality
```
Compare greedy matching vs optimal matching
Gap typically < 5% for Euclidean graphs
```

## Example Execution

For a graph with 4 vertices and odd degrees at vertices B and D:

```
1. Identify odd vertices: {B, D}
2. Find shortest path B→D: distance = 5
3. Create matching: {(B,D)}
4. Augment: Add duplicate edges along B→D path
5. Find Eulerian circuit in augmented graph
6. Total cost = original edges + matching cost
```

## Summary

The Chinese Postman Problem algorithm:
- **Guarantees**: Optimal solution for undirected graphs
- **Key insight**: Convert to Eulerian via minimal augmentation
- **Efficiency**: Polynomial time O(V³)
- **Practical**: Widely applicable to routing problems
- **Extensions**: Directed, mixed, rural variants available

The standard CPP algorithm elegantly combines graph theory (Eulerian circuits), optimization (minimum matching), and algorithms (Hierholzer's) to solve a practical routing problem optimally.