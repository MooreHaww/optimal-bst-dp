"""
experiments.py

Generates test inputs of varying size and probability distribution, and
runs the OBST DP algorithm on each, per the assignment's experimental
analysis requirement (n = 5, 10, 20, 50, 100).

Beyond basic timing, this module also:
  - averages timing over multiple runs to reduce measurement noise
  - reports a hardware-independent operation count (from obst_core)
  - tests multiple probability distributions (random, uniform, skewed)
    to see when the Optimal BST's advantage over a conventional BST is
    largest
  - compares against a conventional (balanced, probability-agnostic) BST
    at every tested size, not just the assignment's single example
  - verifies the DP table's cost against an independent tree-walk at
    every tested size
"""

import random
import statistics
import time
from obst_core import build_optimal_bst, reconstruct_tree
from obst_utils import (
    verify_expected_cost, build_conventional_bst, expected_cost_conventional,
    build_degenerate_bst,
)


# ---------------------------------------------------------------------------
# Input generators
# ---------------------------------------------------------------------------

def generate_random_input(n: int, seed: int = 42):
    """n sorted keys (1..n) with a random probability distribution summing to 1."""
    rng = random.Random(seed + n)
    keys = list(range(1, n + 1))
    raw = [rng.random() + 0.01 for _ in range(n)]
    total = sum(raw)
    probs = [x / total for x in raw]
    probs[-1] += 1.0 - sum(probs)  # fix floating point drift
    return keys, probs


def generate_uniform_input(n: int):
    """n sorted keys (1..n), all equally likely to be searched."""
    keys = list(range(1, n + 1))
    probs = [1.0 / n] * n
    probs[-1] += 1.0 - sum(probs)
    return keys, probs


def generate_skewed_input(n: int, dominant_share: float = 0.9):
    """
    n sorted keys where ONE key (the middle one, arbitrarily) holds
    `dominant_share` of the total probability, and the rest split the
    remainder evenly. Simulates a "hot key" access pattern.
    """
    keys = list(range(1, n + 1))
    if n == 1:
        return keys, [1.0]
    dominant_idx = n // 2
    remainder = (1.0 - dominant_share) / (n - 1)
    probs = [remainder] * n
    probs[dominant_idx] = dominant_share
    probs[-1] += 1.0 - sum(probs)  # fix drift
    return keys, probs


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_single_experiment(n: int, keys, probs, timing_runs: int = 10):
    """
    Runs build_optimal_bst `timing_runs` times on the same input to get a
    stable average execution time (reduces noise from system jitter,
    especially important at small n where a single run is sub-millisecond).
    Also builds both probability-agnostic baselines, times their
    construction, and verifies the DP result against an independent
    tree-walk.

    Construction time is measured for all three trees because section 8 of
    the assignment asks for it "where meaningful". At these sizes the
    wall-clock numbers are small enough to be partly interpreter noise, so
    the degenerate tree's comparison count is reported alongside them: it is
    hardware-independent and exhibits the quadratic build cost directly.
    """
    times = []
    result = None
    for _ in range(timing_runs):
        result = build_optimal_bst(keys, probs)
        times.append(result.build_time_seconds)

    avg_time = statistics.mean(times)
    tree = reconstruct_tree(result)
    verified_cost = verify_expected_cost(tree)
    verification_ok = abs(verified_cost - result.min_cost) < 1e-9

    # Balanced baseline: Theta(n) construction, no comparisons needed.
    balanced_times = []
    for _ in range(timing_runs):
        t0 = time.perf_counter()
        conv_tree = build_conventional_bst(keys, probs)
        balanced_times.append(time.perf_counter() - t0)
    conv_cost = expected_cost_conventional(conv_tree)

    # Degenerate baseline: sequential insertion of sorted keys, Theta(n^2).
    degenerate_times = []
    for _ in range(timing_runs):
        t0 = time.perf_counter()
        deg_tree, deg_comparisons = build_degenerate_bst(keys, probs)
        degenerate_times.append(time.perf_counter() - t0)
    deg_cost = verify_expected_cost(deg_tree)

    improvement_pct = (
        ((conv_cost - result.min_cost) / conv_cost) * 100 if conv_cost > 0 else 0.0
    )
    improvement_vs_degenerate_pct = (
        ((deg_cost - result.min_cost) / deg_cost) * 100 if deg_cost > 0 else 0.0
    )

    return {
        "n": n,
        "avg_time_seconds": avg_time,
        "min_time_seconds": min(times),
        "max_time_seconds": max(times),
        "operation_count": result.operation_count,
        "min_cost": result.min_cost,
        "verified_cost": verified_cost,
        "verification_ok": verification_ok,
        "conventional_cost": conv_cost,
        "conventional_build_seconds": statistics.mean(balanced_times),
        "degenerate_cost": deg_cost,
        "degenerate_build_seconds": statistics.mean(degenerate_times),
        "degenerate_comparisons": deg_comparisons,
        "improvement_pct": improvement_pct,
        "improvement_vs_degenerate_pct": improvement_vs_degenerate_pct,
    }


def run_experiments(sizes=(5, 10, 20, 50, 100), distribution: str = "random",
                     seed: int = 42, timing_runs: int = 10):
    """
    Runs the full experiment suite across `sizes` for a given probability
    distribution ("random", "uniform", or "skewed").
    """
    results = []
    for n in sizes:
        if distribution == "uniform":
            keys, probs = generate_uniform_input(n)
        elif distribution == "skewed":
            keys, probs = generate_skewed_input(n)
        else:
            keys, probs = generate_random_input(n, seed=seed)

        results.append(run_single_experiment(n, keys, probs, timing_runs=timing_runs))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_experiment_table(results, title: str = "Timing / Cost Experiments"):
    """
    Prints two tables:

      1. Cost comparison -- the optimal tree against both probability-agnostic
         baselines (section 8's three-way comparison).
      2. Construction cost -- build time for all three trees plus the
         hardware-independent operation counts, which is what section 9's
         timing table and the complexity discussion are written against.
    """
    print(f"\n=== {title} : expected search cost ===")
    print(f"{'n':>5} | {'Optimal':>9} | {'Verified':>9} | {'Balanced':>9} | "
          f"{'Degenerate':>11} | {'vs Bal.':>8} | {'vs Deg.':>8}")
    print("-" * 78)
    for r in results:
        print(f"{r['n']:>5} | {r['min_cost']:>9.4f} | "
              f"{'OK' if r['verification_ok'] else 'FAIL':>9} | "
              f"{r['conventional_cost']:>9.4f} | {r['degenerate_cost']:>11.4f} | "
              f"{r['improvement_pct']:>7.2f}% | {r['improvement_vs_degenerate_pct']:>7.2f}%")

    print(f"\n=== {title} : construction cost ===")
    print(f"{'n':>5} | {'OBST (s)':>12} | {'Balanced (s)':>13} | {'Degen. (s)':>12} | "
          f"{'OBST ops':>10} | {'Degen. cmp':>11} | Remarks")
    print("-" * 100)
    prev_time = None
    prev_ops = None
    for r in results:
        remark_parts = []
        if prev_time is not None and prev_time > 0:
            remark_parts.append(f"time x{r['avg_time_seconds'] / prev_time:.1f}")
        if prev_ops is not None and prev_ops > 0:
            remark_parts.append(f"ops x{r['operation_count'] / prev_ops:.1f}")
        remark = ", ".join(remark_parts)

        print(f"{r['n']:>5} | {r['avg_time_seconds']:>12.6f} | "
              f"{r['conventional_build_seconds']:>13.6f} | "
              f"{r['degenerate_build_seconds']:>12.6f} | "
              f"{r['operation_count']:>10} | {r['degenerate_comparisons']:>11} | {remark}")

        prev_time = r["avg_time_seconds"]
        prev_ops = r["operation_count"]


def run_all_distributions(sizes=(5, 10, 20, 50, 100), timing_runs: int = 10):
    """
    Convenience function: runs and prints all three distribution types so
    you can directly compare, at each n, how much the Optimal BST's
    advantage over a conventional BST depends on how skewed the search
    probabilities are.
    """
    for dist in ("uniform", "random", "skewed"):
        results = run_experiments(sizes=sizes, distribution=dist, timing_runs=timing_runs)
        print_experiment_table(results, title=f"Distribution: {dist}")


if __name__ == "__main__":
    run_all_distributions()
