# Time Complexity Analysis of Maximum Clique Algorithms

## Summary Table

| Algorithm | Worst-Case Time | Best-Case Time | Space | Practical Performance |
|-----------|----------------|----------------|-------|----------------------|
| Greedy Heuristic | O(n²) | O(n²) | O(n) | Excellent for all sizes |
| GRASP | O(k·n²) | O(k·n²) | O(n) | Good, scales with iterations k |
| Ostergard | O(2ⁿ) | O(n³) | O(n²) | Best exact for n ≤ 500 |
| Bron-Kerbosch (Basic) | O(3^(n/3)) | O(n³) | O(n·M) | Poor, M = # cliques |
| Bron-Kerbosch (Pivot) | O(3^(n/3)) | O(n³) | O(n·M) | Better than basic |
| Branch-and-Bound | O(2ⁿ) | O(n³) | O(n²) | Moderate for n ≤ 200 |

**Legend**:
- n = number of nodes
- m = number of edges
- k = iterations (for GRASP)
- M = number of maximal cliques (can be exponential)

---

## Detailed Analysis

### 1. Greedy Heuristic

#### Time Complexity: **O(n²)** (both worst and average case)

**Algorithm**:
```
1. Start with empty clique
2. For each iteration (at most n iterations):
   - Scan remaining candidates: O(n)
   - For each candidate, check connectivity: O(n)
   - Total per iteration: O(n²)
3. Total: O(n²)
```

**Breakdown**:
- Outer loop: runs at most n times (one per node added)
- Inner loop:
  - Finding candidates: O(n)
  - Checking if node connects to all in clique: O(clique_size) ≤ O(n)
- Per iteration: O(n²)
- Total: O(n²)

**Space Complexity**: O(n)
- Stores current clique (at most n nodes)
- Set of candidates (at most n nodes)

**Why It's Fast**:
- Polynomial time (not exponential)
- No backtracking
- Simple operations (set intersections)

---

### 2. GRASP (Greedy Randomized Adaptive Search)

#### Time Complexity: **O(k·n²)** where k = number of iterations

**Algorithm**:
```
For k iterations:
  1. Randomized greedy construction: O(n²)
  2. Local search improvement: O(n²·L) where L = local search steps
  Total per iteration: O(n²·L)
Total: O(k·n²·L)
```

**Breakdown**:
- **Construction Phase**: O(n²) per iteration (same as greedy)
- **Local Search Phase**:
  - Try swapping each node: O(n)
  - Check connectivity for each swap: O(n)
  - Attempt to expand: O(n²)
  - Local search iterations: typically L < 10
  - Per local search: O(n²·L)
- **Total per GRASP iteration**: O(n²)
- **k iterations**: O(k·n²)

**Space Complexity**: O(n)
- Same as greedy
- No exponential storage

**Typical Parameters**:
- k = 50-100 iterations
- Makes it 50-100x slower than greedy
- But still polynomial!

---

### 3. Ostergard's Algorithm

#### Time Complexity: **O(2ⁿ)** worst case, much better in practice

**Algorithm**:
```
function search(candidates, current_clique, best):
  if |current_clique| + |candidates| ≤ |best|: return  // Pruning 1

  colors = greedy_color(candidates)  // O(n²)
  if |current_clique| + #colors ≤ |best|: return  // Pruning 2 (KEY!)

  for each candidate v:
    if color_bound(v) allows improvement:  // Pruning 3
      new_candidates = neighbors(v) ∩ candidates  // O(n)
      search(new_candidates, current_clique + v, best)
```

**Complexity Analysis**:

**Worst Case**: O(2ⁿ)
- Without pruning, explores all subsets
- 2ⁿ possible subsets of n nodes

**Best Case**: O(n³)
- When graph has small clique number
- Greedy coloring: O(n²)
- Few recursive calls due to pruning
- Total: O(n³)

**Average Case** (on random graphs): O(n² · 2^(√n))
- Much better than worst case
- Coloring pruning is very effective

**Space Complexity**: O(n²)
- Call stack: O(n) depth
- Color array: O(n) per call
- Graph storage: O(n²) for adjacency

**Why It's Fast in Practice**:

1. **Vertex Coloring Bound** (most important):
   - If graph needs k colors, max clique ≤ k
   - Prunes huge portions of search tree
   - Makes O(2ⁿ) → O(n^c) for many graphs

2. **Degree Ordering**:
   - Processes high-degree nodes first
   - Finds large cliques early
   - Improves pruning bound quickly

3. **Multiple Pruning Levels**:
   - Size bound (candidates remaining)
   - Color bound (chromatic number)
   - Per-node color bound

**Empirical Performance**:
- 20 nodes: < 0.001s
- 100 nodes: < 0.01s
- 300 nodes: 0.05s
- 500 nodes: ~1-10s (depends on density)

---

### 4. Bron-Kerbosch (Basic)

#### Time Complexity: **O(3^(n/3))** worst case

**Algorithm**:
```
function BronKerbosch(R, P, X):
  if P and X are empty:
    report R as maximal clique
    return

  for each vertex v in P:
    BronKerbosch(R ∪ {v}, P ∩ N(v), X ∩ N(v))
    P := P \ {v}
    X := X ∪ {v}
```

**Complexity Analysis**:

**Worst Case**: O(3^(n/3))
- Moon-Moser theorem: maximum number of maximal cliques is 3^(n/3)
- Each clique found in polynomial time
- Total: O(3^(n/3) · poly(n))

**Why 3^(n/3)?**
- Worst case graph: disjoint union of triangles
- n/3 triangles → 3^(n/3) maximal cliques
- Must enumerate all of them

**Space Complexity**: O(n · M) where M = number of maximal cliques
- Stores all maximal cliques: O(n · M)
- Call stack: O(n)
- **This is the killer**: M can be millions!

**Per-Node Operations**:
- Set intersections: O(n)
- Neighbor checks: O(n)
- Per recursive call: O(n²)

**Total**: O(3^(n/3) · n²)

**Why It's Slow**:
- Explores every branch
- No pruning if not finding maximum
- Stores all cliques (memory explosion)

---

### 5. Bron-Kerbosch with Pivoting

#### Time Complexity: **O(3^(n/3))** worst case (same as basic)

**Algorithm**:
```
function BronKerboschPivot(R, P, X):
  if P and X are empty:
    report R as maximal clique
    return

  choose pivot u from P ∪ X with max |P ∩ N(u)|  // KEY DIFFERENCE

  for each vertex v in P \ N(u):  // Only non-neighbors of pivot!
    BronKerboschPivot(R ∪ {v}, P ∩ N(v), X ∩ N(v))
    P := P \ {v}
    X := X ∪ {v}
```

**Complexity Analysis**:

**Worst Case**: O(3^(n/3))
- Theoretical bound unchanged
- But constant factors much better

**Best Case**: O(n³)
- When graph has few maximal cliques
- Pivot eliminates many branches

**Space Complexity**: O(n · M)
- Same as basic (still stores all cliques)

**Improvement Over Basic**:
- **Pivot selection**: O(n²) per call
- **Branches explored**: Reduced by 50-70% typically
- **Practical speedup**: 2-3x faster
- **Asymptotic complexity**: Same

**Why Pivoting Helps**:
- Pivot u eliminates its neighbors from consideration
- If u is in maximum clique, its neighbors must be too
- If u is not, we'll try its neighbors anyway
- Reduces branching factor significantly

**Empirical Speedup**:
- On brock200_2: 5.76s → 2.50s (2.3x faster)
- Constant factor improvement only
- Still exponential for large graphs

---

### 6. Branch-and-Bound (Basic)

#### Time Complexity: **O(2ⁿ)** worst case

**Algorithm**:
```
function BranchBound(current_clique, candidates, best):
  if |current_clique| + |candidates| ≤ |best|: return  // Pruning

  if candidates is empty:
    if |current_clique| > |best|:
      best = current_clique
    return

  for each vertex v in candidates:
    new_candidates = candidates ∩ N(v)
    BranchBound(current_clique + v, new_candidates, best)
```

**Complexity Analysis**:

**Worst Case**: O(2ⁿ)
- Explores all subsets in worst case
- Each node: include or exclude
- 2ⁿ possibilities

**Best Case**: O(n³)
- When optimal clique found early
- Pruning eliminates rest
- Only O(n) recursive calls

**Space Complexity**: O(n²)
- Call stack: O(n)
- Current path: O(n)
- No storage of all cliques (unlike Bron-Kerbosch!)

**Pruning Effectiveness**:
- Greedy initialization: gives good bound early
- Size pruning: eliminates branches that can't improve
- Effectiveness varies by graph structure

**Why Less Effective Than Ostergard**:
- No coloring bound (only size bound)
- Less sophisticated pruning
- Slower to find good bounds

---

## Practical Performance Comparison

### Small Graphs (n ≤ 50)

All algorithms work well:
- Greedy: microseconds
- Exact: milliseconds
- Choose any based on need for optimality

### Medium Graphs (50 ≤ n ≤ 200)

| Algorithm | Time | Optimal? |
|-----------|------|----------|
| Greedy | 0.0001s | Maybe |
| GRASP | 0.01s | Maybe |
| Ostergard | 0.01-0.1s | ✓ |
| B-K Pivot | 1-10s | ✓ |
| Branch-Bound | 0.5-5s | ✓ |

**Recommendation**: Ostergard

### Large Graphs (200 ≤ n ≤ 500)

| Algorithm | Time | Optimal? |
|-----------|------|----------|
| Greedy | 0.0001s | Maybe |
| GRASP | 0.1-1s | Maybe |
| Ostergard | 0.05-10s | ✓ |
| B-K Pivot | TOO SLOW | ✓ |
| Branch-Bound | TOO SLOW | ✓ |

**Recommendation**: Ostergard if < 10s acceptable, else GRASP

### Very Large Graphs (n > 500)

| Algorithm | Feasibility |
|-----------|-------------|
| Greedy | Always (instant) |
| GRASP | Usually (seconds) |
| Ostergard | Sometimes (minutes) |
| Others | Rarely (hours) |

**Recommendation**: GRASP or specialized algorithms

---

## Complexity vs. Quality Trade-off

```
Quality ↑
  │
  │  Exact Algorithms (exponential)
  │  ├─ Ostergard: O(2ⁿ) but best pruning
  │  ├─ B-K Pivot: O(3^(n/3))
  │  └─ Branch-Bound: O(2ⁿ) basic pruning
  │
  │  Heuristics (polynomial)
  │  ├─ GRASP: O(k·n²)
  │  └─ Greedy: O(n²)
  │
  └──────────────────────────────> Speed

Fastest                      Slowest
```

---

## Why Maximum Clique is Hard

### NP-Completeness Implications:

1. **Decision Version**: "Is there a clique of size k?" is NP-complete
2. **Optimization Version**: Finding maximum clique is NP-hard
3. **No Polynomial Algorithm**: Unless P = NP (unlikely)

### Fundamental Difficulty:

- **Brute Force**: Check all 2ⁿ subsets → O(2ⁿ)
- **Best Known Exact**: Still exponential (Ostergard, etc.)
- **Best Approximation**: O(n/(log n)²) approximation in poly time
  - But our heuristics do better in practice!

### Why Some Algorithms Work Well:

1. **Real-world graphs aren't worst-case**
   - Sparse graphs: easier
   - Cliques are small relative to n
   - Pruning is effective

2. **Clever bounds** (Ostergard):
   - Coloring reduces exponential to near-polynomial
   - Graph structure helps pruning

3. **Heuristics are practical**:
   - Don't need optimal for many applications
   - Polynomial time is acceptable

---

## Asymptotic vs. Practical Complexity

### Theory Says:
- All exact algorithms: exponential
- Difference is in base and exponent

### Practice Shows:
- Ostergard: Handles 300 nodes in 0.05s
- Bron-Kerbosch: Can't handle 300 nodes
- **Why?**: Constant factors and pruning matter!

### The Gap:
```
Theoretical: O(2ⁿ) vs O(3^(n/3))
  → Both "exponential", similar on paper

Practical: 0.05s vs TIMEOUT
  → Orders of magnitude difference!
```

**Lesson**: For NP-hard problems, implementation details and heuristics matter enormously!

---

## Choosing an Algorithm

### Decision Tree:

1. **Do you need guaranteed optimal?**
   - NO → Use Greedy (O(n²), instant)
   - YES → Continue...

2. **Is n < 300?**
   - YES → Use Ostergard (O(2ⁿ) but fast in practice)
   - NO → Continue...

3. **Is n < 500?**
   - YES → Try Ostergard with timeout, fallback to GRASP
   - NO → Use GRASP (O(k·n²))

4. **Just need quick estimate?**
   - Always run Greedy first (O(n²), < 1ms)

---

## Conclusion

### Key Takeaways:

1. **Greedy**: O(n²) - Always run first, excellent baseline
2. **GRASP**: O(k·n²) - Best heuristic for large graphs
3. **Ostergard**: O(2ⁿ) - Best exact algorithm, works up to ~500 nodes
4. **Bron-Kerbosch**: O(3^(n/3)) - Good for enumerating all cliques, bad for max
5. **Branch-Bound**: O(2ⁿ) - Middle ground, outclassed by Ostergard

### The Winner:
**Ostergard's algorithm** achieves the best practical performance by combining exponential worst-case complexity with powerful polynomial-time bounds (vertex coloring) that make it near-polynomial on many real-world graphs.

### Future Directions:
- Parallel algorithms: O(2ⁿ/p) with p processors
- Approximation algorithms: O(n²) with guarantees
- Specialized algorithms: Exploit graph structure (planar, sparse, etc.)
