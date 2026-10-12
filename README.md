# Optimal Binary Search Tree via Dynamic Programming

Design and Analysis of Algorithms — Project

## Requirements

- Python 3.7 or later (no external libraries required — standard library only)

## Files

| File | Purpose |
|---|---|
| `obst_core.py` | Core DP algorithm: builds the cost table `C` and root table `R`, plus tree reconstruction |
| `obst_utils.py` | Table printing, level/depth computation, cost verification, and the two probability-agnostic baseline BSTs (degenerate and balanced) used for comparison |
| `experiments.py` | Generates test inputs (random, uniform, or skewed probability distributions) and times the algorithm for n = 5, 10, 20, 50, 100, with multi-run averaging, an operation counter, and per-size comparison against a conventional BST |
| `main.py` | Program entry point — ties everything together, handles input (interactive, file, or example), and prints the full report |
| `obst_analysis.py` | **Bonus (not required by the assignment):** brute-force optimality check, extra baselines (greedy, Mehlhorn), entropy bounds, and an empirical complexity fit |
| `sample_input.txt` | Sample input file (the assignment's 5-key example) |
| `sample_output.txt` | Sample output produced by running the program on `sample_input.txt` |

## How to Run

### Option 1 — Interactive menu
```
python3 main.py
```
Shows a six-item menu: run the example input, enter your own input, run the
timing experiments, load input from a file, run the experiments across all
three probability distributions, or run the bonus analysis module.

### Option 2 — Run the assignment's example directly
```
python3 main.py --example
```

### Option 3 — Run the timing experiments (n = 5, 10, 20, 50, 100)
```
python3 main.py --experiments
```
Uses a random probability distribution, averages timing over 10 runs per
size to reduce measurement noise, and reports a hardware-independent
operation count alongside wall-clock time. Also compares against a
conventional BST at every size and verifies the DP cost via an
independent tree-walk.

### Option 4 — Run experiments across all probability distributions
```
python3 main.py --experiments-all
```
Repeats the above for **uniform**, **random**, and **skewed** ("hot key")
probability distributions, so you can see how much the Optimal BST's
advantage over a conventional BST depends on how skewed the search
pattern is. (Spoiler: with uniform probabilities the improvement is ~0%;
with a heavily skewed distribution it grows past 70% as n increases.)

### Option 5 — Provide input from a file
```
python3 main.py --file sample_input.txt
```

### Option 6 — Bonus analysis module
```
python3 main.py --analysis
```
Not required by the assignment. Runs the brute-force optimality check for
n ≤ 8, compares the DP against the greedy and Mehlhorn heuristics plus the
information-theoretic entropy bounds, and fits the measured operation count
against n² and n³ for both the required `classic` method and the optional
`knuth` refinement.

### Controlling table output

By default the DP tables and the tree diagram are printed only when
n ≤ 20 — a 100 × 100 table is unreadable in a terminal and buries the
results. Override this with either flag:
```
python3 main.py --file big_input.txt --show-tables
python3 main.py --example --no-tables
```
The tables are always computed in full regardless; this only controls printing.

## Input File Format

```
<n>
<key_1> <probability_1>
<key_2> <probability_2>
...
<key_n> <probability_n>
```

Keys must be sorted in strictly increasing order. Probabilities must be
non-negative and sum to approximately 1.0 (small floating-point rounding is
tolerated).

Example (`sample_input.txt`):
```
5
10 0.10
20 0.20
30 0.40
40 0.20
50 0.10
```

## What the Program Outputs

For any given input, the program prints:

1. The input keys and probabilities (echoed back, with the probability sum)
2. The full DP cost table `C[i,j]`
3. The full root table `R[i,j]`
4. The reconstructed Optimal BST (as an indented tree diagram)
5. The minimum expected search cost, read directly from `C[1,n]`
6. An independently verified expected cost (computed by walking the actual
   tree and summing `probability × depth`) — this should always match #5
7. The level (depth) of every key in the optimal tree
8. The DP table build time
9. A three-way comparison against two probability-agnostic baseline BSTs
   built over the same keys — a **degenerate** tree (keys inserted one at a
   time in sorted order, which is what a plain BST insertion routine
   produces) and a **balanced** tree (median of each range as its root) —
   showing each tree's structure, expected cost, average depth, and
   construction cost

(At n > 20, items 2, 3, 4 and 7 are summarised rather than printed in full
unless `--show-tables` is given.)

Invalid input is detected before any computation and reported clearly, with
a non-zero exit code. This covers both malformed files (missing file, a
non-numeric key count, fewer key lines than the header promises, a line that
is not `<key> <probability>`, a non-numeric field) and invalid values
(negative probabilities, probabilities not summing to ~1, unsorted or
duplicate keys).

## What the Experiments Show

### Why two baselines

Section 8 of the assignment asks for a comparison against a "conventional
BST". Two different trees can reasonably claim that name, and the program
reports both:

| Baseline | How it is built | Construction cost | What comparing against it shows |
|---|---|---|---|
| **Degenerate** | Insert keys one at a time in the order given | Θ(n²) comparisons — each insert walks the whole right spine, because the input is sorted | The literal reading of "conventional BST". It loses badly, but mostly because it is unbalanced, not because it ignores probabilities |
| **Balanced** | Median key of each range becomes that range's root | Θ(n) | The interesting comparison. This tree is already structurally optimal, so any remaining gap is due **purely** to using the search probabilities |

On the assignment's 5-key example, the optimal tree and the balanced tree
have the *same* average depth (2.2) but different expected costs (1.800 vs
1.900) — the DP does not build a shallower tree, it builds a better-weighted
one.

### How the advantage depends on the distribution

Running `python3 main.py --experiments-all` reveals a pattern worth
discussing in the report:

| Distribution | OBST advantage over the balanced BST |
|---|---|
| Uniform (all keys equally likely) | ~0% — with no probability skew, there's nothing for the DP to optimise; the OBST converges to the same shape as a balanced tree |
| Random | ~4–15%, depending on how the random probabilities happen to fall |
| Skewed (one "hot" key holds 90%) | 59–75%, and **increasing with n** |

This is the core practical argument for using DP-based OBST construction:
the more skewed the real-world access pattern (which is the common case —
e.g. a small number of dictionary words or database keys account for most
lookups), the more expected-cost improvement you get over a naive balanced
tree, and that advantage grows as the key set grows.

### Construction cost

The three trees separate cleanly by build cost, and the measured times track
the theory: doubling n roughly doubles the balanced build (Θ(n)),
quadruples the degenerate build (Θ(n²)), and octuples the OBST build
(Θ(n³)). The OBST's operation count grows by about 7.8× from n = 50 to
n = 100, close to the theoretical 8× for a cubic algorithm. That trade-off
— a cubic preprocessing cost paid once, against a permanently lower expected
search cost — is the practical question the Discussion section should answer.

## Algorithm Summary

- **State:** `C[i,j]` = minimum expected number of comparisons for a BST
  built from keys `i` through `j`.
- **Base case:** `C[i,i] = p[i]`, `C[i,i-1] = 0` (empty subtree).
- **Recurrence:** for `j > i`,
  `C[i,j] = min over k in [i..j] of { C[i,k-1] + C[k+1,j] } + sum(p[i..j])`
- **Root table:** `R[i,j]` records which `k` achieved the minimum, which is
  used to reconstruct the actual tree afterward.
- **Complexity:** Time `Θ(n³)`, Space `Θ(n²)` (three nested loops over `i`,
  `j`, and the choice of root `k`).
