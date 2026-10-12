"""
obst_analysis.py

*** BONUS / EXTENSION MODULE -- not required by the assignment. ***

The assignment specifies the standard Theta(n^3) time / Theta(n^2) space
Dynamic Programming algorithm (obst_core.build_optimal_bst(..., method=
"classic"), the default), and the Complexity Analysis section of the
report is written against that algorithm.

Everything in this file is additional work layered on top of the required
implementation, kept in a separate module so it never gets confused with
the graded submission:

  1. Brute-force optimality check (small n). The tree-walk verification in
     obst_utils only confirms that the reconstructed tree is CONSISTENT with
     C[1,n] -- if the recurrence itself were wrong, both would be wrong in
     the same way. Enumerating every possible BST over n keys and confirming
     the DP result equals the true minimum is a genuinely independent check.

  2. Additional baselines. The median-balanced BST in obst_utils is one
     reasonable probability-agnostic comparison, but two others say more:

       - greedy_root_cost: the heuristic most people's intuition reaches for
         (put the most-likely key at the root, recurse). It is dramatically
         worse than optimal, which is the real argument for why DP is needed.

       - mehlhorn_cost: Mehlhorn's bisection rule, root = the key that most
         evenly splits the subtree's probability weight. Runs in O(n log n)
         and is provably within an additive constant of optimal. Comparing
         against this is the honest question: how much does the expensive
         DP actually buy over a cheap near-optimal heuristic?

  3. Information-theoretic context. Shannon entropy H(p) bounds the optimal
     cost from both sides (roughly H/log2(3) <= cost <= H + 2), which puts
     the DP's answer on an absolute scale rather than only relative to
     whatever baseline happened to be chosen.

  4. Knuth's O(n^2) refinement (obst_core.py, method="knuth"). Not the
     submitted algorithm -- included here only to empirically demonstrate
     that root-monotonicity narrows the search window enough to drop one
     full order of growth, while producing identical costs to the required
     "classic" method. See empirical_complexity_fit() below.
"""

import math
from typing import List, Tuple

from obst_core import build_optimal_bst


# ---------------------------------------------------------------------------
# 1. Brute-force optimality check
# ---------------------------------------------------------------------------

def brute_force_min_cost(probs: List[float]) -> float:
    """
    Exhaustively considers every choice of root at every level -- i.e. every
    one of the Catalan(n) distinct BSTs over the n sorted keys -- and returns
    the minimum expected cost. No memoization, deliberately: this is meant to
    be an independent oracle, not a second dynamic program.

    Exponential. Intended for n <= 10 only.
    """
    n = len(probs)

    def rec(lo: int, hi: int, depth: int) -> float:
        if lo > hi:
            return 0.0
        return min(
            probs[k] * depth + rec(lo, k - 1, depth + 1) + rec(k + 1, hi, depth + 1)
            for k in range(lo, hi + 1)
        )

    return rec(0, n - 1, 1)


def check_dp_against_brute_force(max_n: int = 8, seed: int = 42,
                                 verbose: bool = True) -> bool:
    """
    Runs the DP and the brute-force oracle on random inputs for n = 1..max_n
    and confirms they agree. Returns True if every size matched.
    """
    import random
    all_ok = True
    if verbose:
        print(f"\n=== Brute-force optimality check (n = 1..{max_n}) ===")
        print(f"{'n':>4} | {'DP cost':>12} | {'Brute force':>12} | {'Match':>6}")
        print("-" * 46)

    for n in range(1, max_n + 1):
        rng = random.Random(seed + n)
        raw = [rng.random() + 0.01 for _ in range(n)]
        total = sum(raw)
        probs = [x / total for x in raw]
        probs[-1] += 1.0 - sum(probs)
        keys = list(range(1, n + 1))

        dp = build_optimal_bst(keys, probs).min_cost
        bf = brute_force_min_cost(probs)
        ok = abs(dp - bf) < 1e-12
        all_ok = all_ok and ok
        if verbose:
            print(f"{n:>4} | {dp:>12.8f} | {bf:>12.8f} | {'OK' if ok else 'FAIL':>6}")

    if verbose:
        print(f"\nAll sizes agree: {all_ok}")
    return all_ok


# ---------------------------------------------------------------------------
# 2. Additional baselines
# ---------------------------------------------------------------------------

def greedy_root_cost(probs: List[float]) -> float:
    """
    Greedy heuristic: place the highest-probability key of each range at that
    range's root, then recurse. Intuitive, cheap, and badly suboptimal --
    it ignores that a high-probability key near the middle buys a shallow
    tree, while a high-probability key at an end forces a lopsided one.
    """
    n = len(probs)
    total = 0.0
    stack = [(0, n - 1, 1)]
    while stack:
        lo, hi, depth = stack.pop()
        if lo > hi:
            continue
        k = max(range(lo, hi + 1), key=lambda x: probs[x])
        total += probs[k] * depth
        stack.append((lo, k - 1, depth + 1))
        stack.append((k + 1, hi, depth + 1))
    return total


def mehlhorn_cost(probs: List[float]) -> float:
    """
    Mehlhorn's weight-bisection rule: root each subtree at the key that most
    evenly divides that subtree's total probability weight. O(n log n) with
    binary search on the prefix sums; the loop below is the simple O(n) scan
    per node, which is clearer and fast enough at these sizes.

    Provably within an additive constant of the optimal cost, which is what
    makes it an interesting comparison against the Theta(n^2) / Theta(n^3) DP.
    """
    n = len(probs)
    W = [0.0] * (n + 1)
    for i, p in enumerate(probs):
        W[i + 1] = W[i] + p

    total = 0.0
    stack = [(0, n - 1, 1)]
    while stack:
        lo, hi, depth = stack.pop()
        if lo > hi:
            continue
        best_k, best_diff = lo, float("inf")
        for k in range(lo, hi + 1):
            left_w = W[k] - W[lo]
            right_w = W[hi + 1] - W[k + 1]
            diff = abs(left_w - right_w)
            if diff < best_diff:
                best_diff, best_k = diff, k
        total += probs[best_k] * depth
        stack.append((lo, best_k - 1, depth + 1))
        stack.append((best_k + 1, hi, depth + 1))
    return total


# ---------------------------------------------------------------------------
# 3. Information-theoretic context
# ---------------------------------------------------------------------------

def shannon_entropy(probs: List[float]) -> float:
    """H(p) in bits. Zero-probability keys contribute nothing."""
    return -sum(p * math.log2(p) for p in probs if p > 0)


def entropy_bounds(probs: List[float]) -> Tuple[float, float, float]:
    """
    Returns (H, lower_bound, upper_bound) for the optimal expected cost.

    The optimal number of comparisons is bounded below by H / log2(3)
    (a ternary-outcome comparison yields at most log2(3) bits) and above by
    roughly H + 2 (Mehlhorn's construction achieves this). Useful for saying
    "the DP result is near the information-theoretic floor for this
    distribution" rather than only "it beat the balanced tree by X%".
    """
    H = shannon_entropy(probs)
    return H, H / math.log2(3), H + 2.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def compare_all_strategies(keys: List, probs: List[float], label: str = "") -> dict:
    """Runs the DP plus all baselines on one input and prints a comparison row set."""
    from obst_utils import build_conventional_bst, expected_cost_conventional

    opt = build_optimal_bst(keys, probs).min_cost
    mehl = mehlhorn_cost(probs)
    greedy = greedy_root_cost(probs)
    balanced = expected_cost_conventional(build_conventional_bst(keys, probs))
    H, lo_bound, hi_bound = entropy_bounds(probs)

    if label:
        print(f"\n--- {label} (n = {len(keys)}) ---")
    print(f"  Entropy H(p)                 : {H:.4f} bits")
    print(f"  Information-theoretic range  : {lo_bound:.4f} .. {hi_bound:.4f}")
    print(f"  Optimal BST (DP)             : {opt:.4f}")
    for name, cost in (("Mehlhorn (O(n log n))", mehl),
                       ("Greedy root", greedy),
                       ("Balanced (median)", balanced)):
        gap = (cost - opt) / opt * 100 if opt > 0 else 0.0
        print(f"  {name:<29}: {cost:.4f}   ({gap:+.2f}% vs optimal)")

    return {"optimal": opt, "mehlhorn": mehl, "greedy": greedy,
            "balanced": balanced, "entropy": H}


def empirical_complexity_fit(sizes=(25, 50, 100, 200, 400),
                             method: str = "knuth", seed: int = 42) -> None:
    """
    Fits the measured inner-loop operation count against n^2 and n^3 by
    checking which ratio stays constant as n grows. Stronger evidence for
    the report than raw "time doubled" remarks, since the operation count
    is hardware-independent.
    """
    import random
    print(f"\n=== Empirical complexity fit (method = {method}) ===")
    print(f"{'n':>6} | {'ops':>12} | {'ops / n^2':>11} | {'ops / n^3':>11} | {'ops ratio':>10}")
    print("-" * 64)
    prev = None
    for n in sizes:
        rng = random.Random(seed)
        raw = [rng.random() + 0.01 for _ in range(n)]
        total = sum(raw)
        probs = [x / total for x in raw]
        probs[-1] += 1.0 - sum(probs)
        keys = list(range(1, n + 1))

        r = build_optimal_bst(keys, probs, method=method)
        ops = r.operation_count
        ratio = f"{ops / prev:.2f}x" if prev else "-"
        print(f"{n:>6} | {ops:>12} | {ops / n**2:>11.4f} | {ops / n**3:>11.6f} | {ratio:>10}")
        prev = ops
    print("\nWhichever column stays flat identifies the growth rate.")
    print("Doubling n should roughly quadruple ops for n^2, and octuple it for n^3.")


if __name__ == "__main__":
    from experiments import (generate_random_input, generate_uniform_input,
                             generate_skewed_input)

    check_dp_against_brute_force(max_n=8)

    print("\n\n=== Strategy comparison across distributions ===")
    for gen, name in ((generate_uniform_input, "Uniform"),
                      (generate_random_input, "Random"),
                      (generate_skewed_input, "Skewed (hot key)")):
        keys, probs = gen(100)
        compare_all_strategies(keys, probs, label=name)

    # Required algorithm's own complexity fit (Theta(n^3) confirmation)
    empirical_complexity_fit(sizes=(25, 50, 100, 200), method="classic")

    # Bonus only: same fit for the optional Knuth refinement, for the
    # side-by-side operation-count table in the report's Discussion section
    empirical_complexity_fit(method="knuth")
