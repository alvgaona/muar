# Nearest Neighbor Algorithm for Traveling Salesman Problem

## Overview
The Nearest Neighbor algorithm is a simple greedy heuristic for solving the Traveling Salesman Problem (TSP). It constructs a tour by starting at a given city and repeatedly visiting the nearest unvisited city until all cities have been visited, then returns to the starting city.

## Pseudocode

```
ALGORITHM: NearestNeighbor
INPUT:
    graph G = (V, E, W)  // Complete graph with vertices V, edges E, and weights W
    start_node s ∈ V     // Starting city (optional, default: first node)
    
OUTPUT:
    tour                 // Ordered list of cities forming a Hamiltonian cycle
    total_distance       // Total distance of the tour

BEGIN NearestNeighbor
    // ==========================================
    // INITIALIZATION
    // ==========================================
    IF start_node = NULL THEN
        start_node ← first node in V
    END IF
    
    IF start_node ∉ V THEN
        ERROR "Invalid starting node"
    END IF
    
    tour ← [start_node]
    unvisited ← V \ {start_node}
    current ← start_node
    
    // ==========================================
    // GREEDY CONSTRUCTION
    // ==========================================
    WHILE unvisited ≠ ∅ DO
        // Find the nearest unvisited city
        nearest_node ← NULL
        min_distance ← ∞
        
        FOR EACH node ∈ unvisited DO
            distance ← W(current, node)
            IF distance < min_distance THEN
                min_distance ← distance
                nearest_node ← node
            END IF
        END FOR
        
        // Add nearest node to tour
        tour.append(nearest_node)
        unvisited ← unvisited \ {nearest_node}
        current ← nearest_node
    END WHILE
    
    // ==========================================
    // CALCULATE TOTAL DISTANCE
    // ==========================================
    total_distance ← CalculateTourDistance(tour, W)
    
    RETURN (tour, total_distance)
END NearestNeighbor


// ==========================================
// TOUR DISTANCE CALCULATION
// ==========================================
ALGORITHM: CalculateTourDistance
INPUT:
    tour = [v₁, v₂, ..., vₙ]  // Ordered list of cities
    W                          // Weight/distance function
    
OUTPUT:
    total_distance            // Total tour distance

BEGIN CalculateTourDistance
    total_distance ← 0
    
    FOR i = 1 TO |tour| DO
        from_node ← tour[i]
        to_node ← tour[(i mod |tour|) + 1]  // Wrap around to first city
        total_distance ← total_distance + W(from_node, to_node)
    END FOR
    
    RETURN total_distance
END CalculateTourDistance


// ==========================================
// MULTIPLE STARTS VARIANT
// ==========================================
ALGORITHM: NearestNeighborMultipleStarts
INPUT:
    graph G = (V, E, W)
    num_trials           // Number of different starting points to try
    
OUTPUT:
    best_tour           // Best tour found
    best_distance       // Distance of best tour

BEGIN NearestNeighborMultipleStarts
    IF num_trials = NULL OR num_trials > |V| THEN
        num_trials ← |V|
    END IF
    
    // Randomly select starting nodes
    start_nodes ← RandomSample(V, num_trials)
    
    best_tour ← NULL
    best_distance ← ∞
    
    // Try NN from each starting node
    FOR EACH start_node ∈ start_nodes DO
        (tour, distance) ← NearestNeighbor(G, start_node)
        
        IF distance < best_distance THEN
            best_distance ← distance
            best_tour ← tour
        END IF
    END FOR
    
    RETURN (best_tour, best_distance)
END NearestNeighborMultipleStarts
```

## Algorithm Components

### 1. Core Strategy
- **Greedy Selection**: Always choose the nearest unvisited city
- **Construction**: Build tour incrementally, one city at a time
- **Completion**: Return to start city after visiting all cities

### 2. Key Data Structures
- **Tour**: Ordered list of cities representing the path
- **Unvisited Set**: Tracks cities not yet added to tour
- **Current Node**: The last city added to the tour

### 3. Distance Calculation
- Sum of edge weights along the tour
- Includes edge from last city back to start (completing the cycle)

## Time Complexity

| Operation | Complexity |
|-----------|------------|
| Single Run | O(n²) |
| Multiple Starts (k trials) | O(k × n²) |
| All Starts | O(n³) |

### Detailed Analysis:
- **Finding nearest city**: O(n) for each city
- **Building complete tour**: n iterations × O(n) = O(n²)
- **Distance calculation**: O(n)
- **Total**: O(n²)

## Space Complexity

- Tour storage: O(n)
- Unvisited set: O(n)
- Distance matrix (if precomputed): O(n²)
- **Total**: O(n) without precomputation, O(n²) with distance matrix

## Advantages

1. **Simple to implement** - straightforward greedy logic
2. **Fast execution** - O(n²) is polynomial time
3. **Intuitive** - mirrors human problem-solving approach
4. **Memory efficient** - O(n) space for basic version
5. **Always produces a valid tour** - guaranteed to visit all cities exactly once
6. **Good starting point** for improvement heuristics

## Disadvantages

1. **No quality guarantee** - can be arbitrarily bad compared to optimal
2. **Myopic decisions** - doesn't consider global tour structure
3. **Starting point dependent** - different starts yield different tours
4. **Suboptimal** - typically 15-30% above optimal
5. **No backtracking** - can't undo poor early choices

## Quality Analysis

### Worst Case
- No bounded approximation ratio
- Can be Θ(log n) times optimal in worst case
- Example: Cities arranged in a line where NN zigzags

### Average Case
- Typically 15-30% above optimal for random instances
- Better performance on clustered data
- Worse on structured problems (grids, circles)

### Improvement Strategies
1. **Multiple starts**: Try different starting cities
2. **Randomization**: Add randomness to selection
3. **Look-ahead**: Consider k-nearest neighbors
4. **Post-optimization**: Apply 2-opt, 3-opt, etc.

## Variants and Extensions

### 1. Farthest Insertion
```
Instead of nearest unvisited:
    Select city that maximizes minimum distance to tour
```

### 2. Cheapest Insertion
```
Instead of nearest from current:
    Select city that minimizes tour increase when inserted
```

### 3. Random Nearest Neighbor
```
Instead of always nearest:
    Select from k-nearest with probability
```

### 4. Savings Algorithm (Clarke-Wright)
```
Start with radial tours from depot
Merge tours based on savings
```

## Practical Considerations

### 1. Distance Matrix Precomputation
```python
# Precompute for O(1) distance lookups
distance_matrix[i][j] = distance(city_i, city_j)
```
- Trade space for time
- Worthwhile for multiple runs
- Essential for large problems

### 2. Starting City Selection
- **Random**: Unbiased exploration
- **Centroid**: Start from geometric center
- **Peripheral**: Start from boundary city
- **All cities**: Try each as start (O(n³) total)

### 3. Tie Breaking
When multiple cities have same distance:
- **First found**: Deterministic, fast
- **Random choice**: Adds diversity
- **Secondary criteria**: Use angle, coordinates, etc.

### 4. Parallelization
- Multiple starts are independent
- Easily parallelizable across cores
- Linear speedup possible

## Implementation Tips

### 1. Efficient Nearest Search
```
# Use priority queue for large n
min_heap = [(distance(current, city), city) for city in unvisited]
heapify(min_heap)
nearest = heappop(min_heap)[1]
```

### 2. Early Termination
```
# For approximate solutions
IF time_limit_exceeded OR good_enough_solution THEN
    RETURN current_best
```

### 3. Hybrid Approaches
```
# Use NN for initial solution
initial_tour = nearest_neighbor(graph)
# Apply improvement heuristic
improved_tour = lin_kernighan(initial_tour)
```

## Example Applications

1. **Delivery Route Planning**: Quick route generation for daily deliveries
2. **PCB Drilling**: Tool path optimization in manufacturing
3. **DNA Sequencing**: Fragment assembly ordering
4. **Tournament Scheduling**: Minimize travel for sports teams
5. **Network Design**: Initial topology for optimization

## Performance Comparison

| Algorithm | Time | Quality (% above optimal) | Use Case |
|-----------|------|---------------------------|----------|
| Nearest Neighbor | O(n²) | 15-30% | Quick approximation |
| NN + 2-opt | O(n²) | 5-15% | Better quality, still fast |
| NN Multiple Starts | O(n³) | 10-20% | Improved without complexity |
| Christofides | O(n³) | ≤50% guaranteed | Theoretical guarantee |
| Lin-Kernighan | O(n²·²) | 0-5% | Near-optimal solutions |

## Pseudocode for Common Improvements

### 2-Opt Post-Processing
```
ALGORITHM: Improve_NN_With_2Opt
INPUT: tour from NN
OUTPUT: improved_tour

improved ← TRUE
WHILE improved DO
    improved ← FALSE
    FOR i = 1 TO |tour| - 2 DO
        FOR j = i + 2 TO |tour| DO
            IF cost_of_swap(i, j) < 0 THEN
                reverse_tour_segment(i+1, j)
                improved ← TRUE
                BREAK
            END IF
        END FOR
    END FOR
END WHILE
RETURN tour
```

## Summary

The Nearest Neighbor algorithm represents the simplest reasonable approach to TSP:
- **Pros**: Fast, simple, always valid
- **Cons**: Suboptimal, myopic, variable quality
- **Best for**: Initial solutions, time-critical applications, small problems
- **Not for**: When optimality is required, highly structured problems

It serves as an excellent baseline and starting point for more sophisticated TSP algorithms.