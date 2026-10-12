"""
obst_core.py

Core Dynamic Programming algorithm for the Optimal Binary Search Tree (OBST)
problem, following the recurrence:

    C[i, i]   = p[i]
    C[i, j]   = min over k in [i..j] of { C[i, k-1] + C[k+1, j] } + sum(p[i..j])

where C[i, j] is the minimum expected number of comparisons for a BST built
from keys i..j (1-indexed, inclusive), and R[i, j] records which key was
chosen as the root of that optimal subtree.

Internally we use 1-based indexing to match the standard textbook formulation
(Levitin, Ch. 8), with sentinel row/column 0 and n+1 to represent empty
subtrees cleanly.

Two fill strategies are provided:

  method="classic"  -- the textbook Theta(n^3) triple loop, k ranging over
                       the full interval [i..j].

  method="knuth"    -- Knuth's (1971) optimization. The optimal root is
                       monotone in both arguments:

                           R[i][j-1]  <=  R[i][j]  <=  R[i+1][j]

                       so k need only range over that narrower window. Both
                       bounds live on the previous diagonal and are therefore
                       already filled. The window widths telescope, giving
                       Theta(n^2) total time. The optimization is valid here
                       because the weight function w(i,j) = sum(p[i..j]) is
                       non-negative and monotone, hence satisfies the
                       quadrangle inequality.

Both methods produce identical cost tables (verified to 1e-12 across uniform,
random and skewed distributions for n up to 100).

Subtree weights come from a prefix-sum array W, so sum(p[i..j]) is an O(1)
lookup rather than an O(n) re-summation per cell. This matters for both
methods, and is a prerequisite for Knuth's optimization actually reaching
n^2 -- an O(n) weight lookup would drag it back toward n^3.
"""

from dataclasses import dataclass
from typing import List, Optional, Any
import time


@dataclass
class OBSTResult:
    keys: List           # original keys, 0-indexed for convenience
    probs: List[float]
    C: List[List[float]] # cost table, 1-indexed access via C[i][j]
    R: List[List[int]]   # root table, 1-indexed access via R[i][j]
    min_cost: float
    build_time_seconds: float
    operation_count: int = 0  # number of inner-loop "k" evaluations (hardware-independent cost measure)
    method: str = "classic"   # which fill strategy produced this result


class TreeNode:
    """Simple binary tree node used to represent the reconstructed OBST."""
    __slots__ = ("key", "prob", "left", "right")

    def __init__(self, key, prob):
        self.key = key
        self.prob = prob
        self.left: Optional["TreeNode"] = None
        self.right: Optional["TreeNode"] = None


def validate_input(keys: List, probs: List[float], tol: float = 1e-6) -> None:
    """
    Validates the inputs per the assignment spec:
      - keys must be sorted, no duplicates
      - probabilities must be non-negative
      - probabilities must sum to ~1 (allowing floating point rounding)
    Raises ValueError with a descriptive message if invalid.
    """
    if len(keys) == 0:
        raise ValueError("At least one key is required.")
    if len(keys) != len(probs):
        raise ValueError(
            f"Number of keys ({len(keys)}) must match number of probabilities ({len(probs)})."
        )
    if any(keys[i] >= keys[i + 1] for i in range(len(keys) - 1)):
        raise ValueError("Keys must be sorted in strictly increasing order (no duplicates).")
    if any(p < 0 for p in probs):
        raise ValueError("All probabilities must be non-negative.")
    total = sum(probs)
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Probabilities must sum to approximately 1.0 (got {total:.6f})."
        )


def build_optimal_bst(keys: List, probs: List[float],
                      method: str = "classic") -> OBSTResult:
    """
    Builds the DP cost table C and root table R for the OBST problem.

    Uses 1-based indexing internally (arrays sized n+2) to mirror the
    textbook recurrence exactly:
        C[i, i-1] = 0            (empty subtree, base case)
        C[i, i]   = p[i]         (single key)
        C[i, j]   = min_k { C[i,k-1] + C[k+1,j] } + sum(p[i..j])   for j > i

    method:
        "classic" -> Theta(n^3), full k range (default). This is the
                     algorithm the assignment specifies and the one the
                     Complexity Analysis section should be written
                     against -- three nested loops over i, j, and the
                     choice of root k.
        "knuth"   -> Theta(n^2), root-monotonicity window. An OPTIONAL
                     extension beyond the assignment's requirements; see
                     obst_analysis.py for the empirical operation-count
                     comparison against "classic". Do not use this as
                     the basis for the required O(n^3) complexity writeup.

    Returns an OBSTResult containing both tables plus the minimum cost
    for the whole set of keys and the time taken to build the tables.
    """
    if method not in ("knuth", "classic"):
        raise ValueError(f"method must be 'knuth' or 'classic' (got {method!r}).")

    validate_input(keys, probs)
    n = len(keys)

    # 1-indexed probability array: P[1..n]
    P = [0.0] + list(probs)

    # Prefix sums so that sum(p[i..j]) == W[j] - W[i-1] in O(1).
    W = [0.0] * (n + 2)
    for i in range(1, n + 1):
        W[i] = W[i - 1] + P[i]

    # C and R sized (n+2) x (n+2) so we can index rows/cols 0..n+1 safely
    C = [[0.0] * (n + 2) for _ in range(n + 2)]
    R = [[0] * (n + 2) for _ in range(n + 2)]

    start = time.perf_counter()
    op_count = 0  # counts each evaluation of C[i,k-1] + C[k+1,j] in the innermost loop

    # Base cases: single-key subtrees and empty subtrees
    for i in range(1, n + 1):
        C[i][i - 1] = 0.0   # empty subtree to the left of key i
        C[i][i] = P[i]
        R[i][i] = i
    C[n + 1][n] = 0.0       # empty subtree to the right of key n

    use_knuth = (method == "knuth")

    # Fill by increasing "diagonal" d = j - i, exactly as in the slides
    for d in range(1, n):
        for i in range(1, n - d + 1):
            j = i + d

            if use_knuth:
                # Both bounds sit on diagonal d-1, already computed.
                k_lo = R[i][j - 1]
                k_hi = R[i + 1][j]
            else:
                k_lo, k_hi = i, j

            min_val = float("inf")
            k_min = k_lo
            row_i = C[i]                      # local alias, avoids repeated lookup
            for k in range(k_lo, k_hi + 1):
                cost = row_i[k - 1] + C[k + 1][j]
                op_count += 1
                if cost < min_val:
                    min_val = cost
                    k_min = k

            R[i][j] = k_min
            C[i][j] = min_val + (W[j] - W[i - 1])   # O(1) subtree weight

    elapsed = time.perf_counter() - start

    return OBSTResult(
        keys=list(keys),
        probs=list(probs),
        C=C,
        R=R,
        min_cost=C[1][n],
        build_time_seconds=elapsed,
        operation_count=op_count,
        method=method,
    )


def reconstruct_tree(result: OBSTResult) -> Optional[TreeNode]:
    """
    Walks the root table R to rebuild the actual optimal BST as a
    TreeNode structure. Keys/probs are pulled from result.keys/result.probs
    (0-indexed), while R is 1-indexed.

    Iterative (explicit stack) rather than recursive: for heavily skewed
    probability distributions the optimal tree is close to degenerate, so
    a recursive rebuild would hit Python's recursion limit for large n.
    """
    n = len(result.keys)
    if n == 0:
        return None

    R = result.R
    keys = result.keys
    probs = result.probs

    def make(idx_1based: int) -> TreeNode:
        return TreeNode(keys[idx_1based - 1], probs[idx_1based - 1])

    root_idx = R[1][n]
    root = make(root_idx)

    # stack entries: (parent_node, attach_side, i, j)
    stack = [(root, "left", 1, root_idx - 1),
             (root, "right", root_idx + 1, n)]

    while stack:
        parent, side, i, j = stack.pop()
        if i > j:
            continue
        k = R[i][j]
        node = make(k)
        setattr(parent, side, node)
        stack.append((node, "left", i, k - 1))
        stack.append((node, "right", k + 1, j))

    return root
