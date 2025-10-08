# GRASP Algorithm for Maximum Clique Problem

## Overview
GRASP (Greedy Randomized Adaptive Search Procedure) is a metaheuristic that combines a greedy randomized construction phase with a local search phase to find high-quality solutions to the maximum clique problem.

## Pseudocode

```
ALGORITHM: GRASP-MaxClique
INPUT: 
    graph G = (V, E)  // Graph represented as adjacency list
    iterations        // Number of GRASP iterations
    α ∈ [0, 1]       // Randomization parameter (0=pure greedy, 1=pure random)
    
OUTPUT: 
    best_clique      // Best clique found

BEGIN GRASP-MaxClique
    best_clique ← ∅
    
    FOR iteration = 1 TO iterations DO
        // ==========================================
        // PHASE 1: CONSTRUCTION (Randomized Greedy)
        // ==========================================
        clique ← ∅
        candidates ← V  // All vertices are initial candidates
        
        WHILE candidates ≠ ∅ DO
            // Calculate degree of each candidate in induced subgraph
            degrees ← []
            FOR EACH node ∈ candidates DO
                degree ← |{neighbor ∈ candidates : (node, neighbor) ∈ E}|
                degrees.append((degree, node))
            END FOR
            
            IF degrees = ∅ THEN
                BREAK
            END IF
            
            // Sort candidates by degree (descending)
            degrees.sort(reverse=TRUE)
            
            // Build Restricted Candidate List (RCL)
            // RCL contains top α% of candidates
            rcl_size ← max(1, ⌊|degrees| × α⌋)
            RCL ← {node : (degree, node) ∈ degrees[1..rcl_size]}
            
            // Randomly select from RCL
            selected ← random_choice(RCL)
            clique ← clique ∪ {selected}
            
            // Update candidates: keep only neighbors of selected node
            candidates ← candidates ∩ Neighbors(selected)
        END WHILE
        
        // ==========================================
        // PHASE 2: LOCAL SEARCH (Improvement)
        // ==========================================
        clique ← LocalSearch(graph, clique)
        
        // Update best solution
        IF |clique| > |best_clique| THEN
            best_clique ← clique
        END IF
    END FOR
    
    RETURN best_clique
END GRASP-MaxClique


// ==========================================
// LOCAL SEARCH SUBROUTINE
// ==========================================
ALGORITHM: LocalSearch
INPUT: 
    graph G = (V, E)
    initial_clique
    
OUTPUT: 
    improved_clique

BEGIN LocalSearch
    clique ← initial_clique
    improved ← TRUE
    
    WHILE improved = TRUE DO
        improved ← FALSE
        
        // Try 1-1 swaps: remove one node, add one node
        FOR EACH node_out ∈ clique DO
            remaining ← clique \ {node_out}
            candidates ← V \ clique
            
            FOR EACH node_in ∈ candidates DO
                // Check if node_in is connected to all remaining nodes
                IF node_in is adjacent to all nodes in remaining THEN
                    new_clique ← remaining ∪ {node_in}
                    
                    // Try to expand the new clique
                    expanded ← new_clique
                    FOR EACH extra ∈ (candidates \ {node_in}) DO
                        IF extra is adjacent to all nodes in expanded THEN
                            expanded ← expanded ∪ {extra}
                        END IF
                    END FOR
                    
                    // Accept improvement if expansion yields larger clique
                    IF |expanded| > |clique| THEN
                        clique ← expanded
                        improved ← TRUE
                        BREAK
                    END IF
                END IF
            END FOR
            
            IF improved = TRUE THEN
                BREAK
            END IF
        END FOR
    END WHILE
    
    RETURN clique
END LocalSearch
```

## Algorithm Components

### 1. Construction Phase
- **Greedy Component**: Candidates are evaluated based on their degree in the induced subgraph
- **Randomization**: Instead of always selecting the best candidate, we:
  1. Create a Restricted Candidate List (RCL) with the top α% candidates
  2. Randomly select from the RCL
- **Adaptive**: The candidate list adapts as we build the clique (only neighbors of selected nodes remain)

### 2. Local Search Phase
- **Neighborhood**: 1-1 swaps (remove one node, add another)
- **Improvement Strategy**: 
  1. Try swapping each node in the current clique
  2. After each valid swap, attempt to expand the clique greedily
  3. Accept the swap if it leads to a larger clique
- **Termination**: When no improving swap can be found

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `iterations` | Number of GRASP iterations | 50-500 |
| `α` | Randomization factor | 0.1-0.3 |
| | α = 0: Pure greedy (deterministic) | |
| | α = 1: Pure random selection | |
| | α ∈ (0,1): Balance between greedy and random | |

## Time Complexity

- **Construction Phase**: O(n²) per iteration
  - Building RCL: O(n log n) for sorting
  - Updating candidates: O(n)
  
- **Local Search Phase**: O(n³) worst case per iteration
  - For each node in clique: O(n)
  - For each candidate replacement: O(n)
  - Checking adjacency and expansion: O(n)

- **Overall**: O(k × n³) where k = number of iterations

## Space Complexity

- O(n) for storing the graph adjacency list
- O(n) for storing cliques and candidate sets
- **Total**: O(n)

## Advantages

1. **Balances exploration and exploitation** through randomization parameter α
2. **Simple to implement** and understand
3. **Memory efficient** - only stores best solution
4. **Parallelizable** - iterations are independent
5. **Good empirical performance** on many graph instances

## Disadvantages

1. **No optimality guarantee** - heuristic approach
2. **Performance depends on parameter tuning** (α, iterations)
3. **May get stuck in local optima** despite randomization
4. **Runtime increases linearly with iterations**

## Practical Considerations

- Start with α ≈ 0.2-0.3 for good balance
- Increase iterations for better quality at cost of time
- Consider problem-specific local search operators
- Can be combined with other techniques (e.g., path relinking, reactive GRASP)
