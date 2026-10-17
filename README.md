<div align="center">

# 🌳 Optimal Binary Search Tree

**Build the search tree that needs the fewest comparisons on average, using dynamic programming, and measure how much it really helps.**

![Python](https://img.shields.io/badge/python-3.7%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-success?style=for-the-badge)
![Complexity](https://img.shields.io/badge/time-%CE%98(n%C2%B3)-orange?style=for-the-badge)
![Course](https://img.shields.io/badge/course-Algorithms-8A2BE2?style=for-the-badge)

</div>

---

## 💡 Why this project exists

A normal binary search tree treats every key the same. Real lookups do not: in a dictionary or a database a few keys are searched all the time and most almost never. If you know how often each key is searched, you can arrange the tree so the popular keys sit near the top and the average search gets cheaper.

Finding that best arrangement is a classic **dynamic programming** problem. This is my **Design and Analysis of Algorithms** project: implement it, prove it is correct, and measure whether it is worth the cost.

## 🔎 What it does

Given sorted keys and the probability of searching for each, the program:

1. Builds a **cost table** `C[i,j]` and a **root table** `R[i,j]` with dynamic programming.
2. Reconstructs the optimal tree and prints it as a diagram.
3. Reports the minimum expected search cost and **verifies it independently** by walking the tree.
4. Compares the result with two ordinary trees: a **degenerate** one (plain insertion of sorted keys) and a **balanced** one (median as root).
5. Runs timing experiments for n = 5, 10, 20, 50 and 100 on uniform, random and skewed probabilities.
6. Rejects bad input (unsorted or duplicate keys, negative probabilities, probabilities that do not sum to 1) with a clear message and a non-zero exit code.

On the assignment's 5-key example:

```text
30 (p=0.4000)
    L-- 20 (p=0.2000)
    |   L-- 10 (p=0.1000)
    R-- 40 (p=0.2000)
        R-- 50 (p=0.1000)

Optimal (DP)            expected cost 1.800
Balanced (median root)  expected cost 1.900
Degenerate (insertion)  expected cost 3.000
```

### 📊 The main finding: it depends on how skewed the searches are

| Probabilities | Optimal BST vs balanced BST (n = 100) |
|---|---|
| Uniform (every key equally likely) | about 0%: nothing to optimise |
| Random | about 12% better |
| Skewed (one "hot" key holds 90%) | about **75% better**, and the gap grows with n |

The cost of building the tree follows the theory: the operation count grows about 7.8× when n doubles from 50 to 100, close to the 8× expected for a cubic algorithm. The trade-off is a one-time cubic build for a permanently cheaper search.

*Numbers come from `python main.py --experiments-all`; random inputs vary slightly between runs.*

## ⚙️ Setup

You need Python 3.7 or newer. No packages to install.

```bash
git clone https://github.com/MooreHaww/optimal-bst-dp.git
cd optimal-bst-dp
python main.py --example
```

| Command | What it does |
|---|---|
| `python main.py` | Interactive menu |
| `python main.py --example` | The assignment's 5-key example |
| `python main.py --file sample_input.txt` | Read keys and probabilities from a file |
| `python main.py --experiments` | Timing experiments, n = 5 to 100 |
| `python main.py --experiments-all` | Experiments for uniform, random and skewed probabilities |
| `python main.py --analysis` | Bonus: brute-force check, heuristics, entropy bounds, complexity fit |
| `--show-tables` / `--no-tables` | Force or hide the DP tables (shown by default only when n ≤ 20) |

Input file format (keys strictly increasing, probabilities sum to 1):

```text
5
10 0.10
20 0.20
30 0.40
40 0.20
50 0.10
```

## 🎯 What I wanted to learn

- How to turn a problem into a **DP recurrence**: define the state, the base case, and how to combine sub-answers.
- How to rebuild the actual answer (the tree) from a table of choices, not just its cost.
- How to check an algorithm two ways: an independent tree-walk, and a brute-force search for small n.
- How to measure complexity honestly, with a hardware-independent operation count next to wall-clock time.
- Why a fair comparison matters: beating a degenerate tree proves little, beating a balanced tree proves the DP is using the probabilities.

## 🏗️ How it was built

| File | Purpose |
|---|---|
| `obst_core.py` | The DP: cost table, root table, tree reconstruction |
| `obst_utils.py` | Table and tree printing, depth computation, cost verification, the two baseline trees |
| `experiments.py` | Input generators and the timing and comparison runs |
| `main.py` | Entry point: menu, flags, file input, full report |
| `obst_analysis.py` | Bonus module: brute force, greedy and Mehlhorn heuristics, entropy bounds, curve fitting |
| `sample_input.txt`, `sample_output.txt` | The 5-key example and its output |

**The algorithm**

- `C[i,j]` = minimum expected comparisons for a tree over keys `i..j`.
- Base case: `C[i,i] = p[i]` and `C[i,i-1] = 0` (an empty tree).
- Recurrence: `C[i,j] = min over k in [i..j] of ( C[i,k-1] + C[k+1,j] ) + sum(p[i..j])`.
- `R[i,j]` stores the best root `k`, which is used to rebuild the tree.
- Time Θ(n³), space Θ(n²).

**Build order**

1. Core DP and tree reconstruction.
2. Input validation and the report format.
3. Cost verification by walking the tree.
4. Baseline trees and the comparison table.
5. Experiments with several distributions and an operation counter.
6. Bonus analysis module.

## 🗺️ Plan / what's next

- [x] Optimal BST by DP with verification
- [x] Comparison with degenerate and balanced trees
- [x] Experiments across three distributions
- [x] Bonus analysis (brute force, heuristics, entropy bounds)
- [ ] Make Knuth's optimisation (already compared in the bonus analysis) the default, which brings the time down to about O(n²)
- [ ] Add automated unit tests
- [ ] Plot the experiment results instead of printing tables
- [ ] Visualise the tree as an image

## 📄 Notes

Built for a university Design and Analysis of Algorithms course.
