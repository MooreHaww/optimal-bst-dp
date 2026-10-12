"""
obst_utils.py

Supporting utilities for the OBST project:
  - printing the DP cost table and root table
  - computing the level/depth of each key in a reconstructed tree
  - verifying the expected cost by walking the actual tree (independent
    check against the DP table's C[1,n] value)
  - building the two probability-agnostic baseline trees used in the
    comparison section of the report:
      * a DEGENERATE BST, produced by inserting the keys one at a time in
        the order given (the literal "conventional BST" of the assignment);
      * a BALANCED BST, produced by taking the median key of each range as
        its root.

Why two baselines rather than one:

The assignment (section 8) asks for a comparison against "a conventional
BST built using the same keys". Read literally, the conventional way to
build a BST is sequential insertion -- and because the input keys are
required to be sorted, that degenerates into a right-leaning chain with
expected cost Theta(n). That is the honest literal reading and it is
included.

But it is also a weak opponent: it loses because it is unbalanced, not
because it ignores probabilities, so beating it says nothing about what
the DP actually buys. The median-balanced tree isolates that question --
it is structurally as good as a BST can be while still knowing nothing
about the search probabilities, so any remaining gap between it and the
optimal tree is attributable purely to the probability information.

Reporting both gives the three-way comparison (degenerate / balanced /
optimal) that separates the two effects.

All tree walks here are iterative. The optimal tree is shallow in
practice, but the degenerate baseline has depth exactly n, so a recursive
walk over it would hit Python's recursion limit for large inputs. Keeping
every walker iterative also keeps this module consistent with
obst_core.reconstruct_tree, which is iterative for the same reason.
"""

from typing import List, Optional, Dict, Tuple
from obst_core import TreeNode, OBSTResult


def print_cost_table(result: OBSTResult) -> None:
    n = len(result.keys)
    C = result.C
    print("\nDP Cost Table C[i,j]  (rows = i, cols = j)")
    header = "     " + "".join(f"{j:>8}" for j in range(0, n + 1))
    print(header)
    for i in range(1, n + 1):
        row = f"i={i:<3}"
        for j in range(0, n + 1):
            if j < i - 1:
                row += " " * 8
            else:
                row += f"{C[i][j]:>8.3f}"
        print(row)


def print_root_table(result: OBSTResult) -> None:
    n = len(result.keys)
    R = result.R
    print("\nRoot Table R[i,j]  (rows = i, cols = j)  -- values are 1-based key indices")
    header = "     " + "".join(f"{j:>8}" for j in range(1, n + 1))
    print(header)
    for i in range(1, n + 1):
        row = f"i={i:<3}"
        for j in range(1, n + 1):
            if j < i:
                row += " " * 8
            else:
                row += f"{R[i][j]:>8}"
        print(row)


# ---------------------------------------------------------------------------
# Tree walks (generic -- work on any node exposing .key/.prob/.left/.right)
# ---------------------------------------------------------------------------

def compute_levels(root) -> Dict:
    """
    Returns a dict mapping key -> level, with the root at level 1 (matching
    the "1 comparison at the root" convention used in the slides' formula
    level(a_s) + 1).

    Iterative: the degenerate baseline tree has depth n.
    """
    levels = {}
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if node is None:
            continue
        levels[node.key] = depth
        stack.append((node.left, depth + 1))
        stack.append((node.right, depth + 1))
    return levels


def verify_expected_cost(root) -> float:
    """
    Independently computes the expected search cost by walking the actual
    tree: sum of p(key) * level(key). For the optimal tree this should match
    C[1,n] from the DP table -- it is the "verify the resulting expected cost
    using the constructed tree" requirement in the assignment. It is also
    used to cost the two baseline trees.
    """
    total = 0.0
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if node is None:
            continue
        total += node.prob * depth
        stack.append((node.left, depth + 1))
        stack.append((node.right, depth + 1))
    return total


def average_depth(root) -> float:
    """Mean level over all keys (unweighted by probability)."""
    levels = compute_levels(root)
    return sum(levels.values()) / len(levels) if levels else 0.0


def print_tree(root, max_nodes: int = 200) -> None:
    """
    Indented visual of the tree structure. The root is printed without a
    child-branch marker; every other node is tagged L-- or R-- according to
    which child it is of its parent.

    Iterative, and capped at max_nodes lines so that printing a degenerate
    100-key chain does not swamp the terminal.
    """
    if root is None:
        return
    printed = 0
    # stack entries: (node, prefix, marker)
    stack = [(root, "", "")]
    while stack:
        node, prefix, marker = stack.pop()
        if node is None:
            continue
        if printed >= max_nodes:
            print(prefix + "... (output truncated)")
            return
        print(f"{prefix}{marker}{node.key} (p={node.prob:.4f})")
        printed += 1
        child_prefix = prefix + ("    " if not marker else ("|   " if marker.startswith("L") else "    "))
        # push right first so left is popped (and printed) first
        stack.append((node.right, child_prefix, "R-- "))
        stack.append((node.left, child_prefix, "L-- "))


# ---------------------------------------------------------------------------
# Baseline BSTs for the required comparison (section 8)
# ---------------------------------------------------------------------------

class ConventionalNode:
    """Node type for both probability-agnostic baseline trees."""
    __slots__ = ("key", "prob", "left", "right")

    def __init__(self, key, prob):
        self.key = key
        self.prob = prob
        self.left = None
        self.right = None


def build_degenerate_bst(keys: List, probs: List[float]) -> Tuple[Optional[ConventionalNode], int]:
    """
    The literal "conventional BST": insert each key into an initially empty
    BST, one at a time, in the order supplied. No balancing, no knowledge of
    probabilities.

    Because the assignment requires the input keys to be sorted, every key
    is larger than everything already in the tree, so each insertion walks
    the entire existing right spine before attaching at the bottom. The
    result is a right-leaning chain of depth n -- a linked list with extra
    steps -- and the build itself costs 0 + 1 + ... + (n-1) = Theta(n^2)
    comparisons rather than the Theta(n log n) that insertion into a BST
    costs on randomly ordered input.

    Returns (root, comparison_count). The comparison count is returned
    because it is the meaningful construction-cost measure for section 8:
    wall-clock time at n <= 100 is dominated by interpreter overhead, while
    the comparison count is hardware-independent and shows the quadratic
    growth directly.
    """
    root = None
    comparisons = 0
    for key, prob in zip(keys, probs):
        node = ConventionalNode(key, prob)
        if root is None:
            root = node
            continue
        current = root
        while True:
            comparisons += 1
            if key < current.key:
                if current.left is None:
                    current.left = node
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = node
                    break
                current = current.right
    return root, comparisons


def build_conventional_bst(keys: List, probs: List[float]) -> Optional[ConventionalNode]:
    """
    The balanced probability-agnostic baseline: recursively take the middle
    key of the current range as that range's root (the classic "build a
    balanced BST from a sorted array" construction).

    This is the stronger of the two baselines. It is structurally optimal --
    no BST over n keys can have smaller height -- while still knowing
    nothing about the search probabilities, so the gap between this tree and
    the optimal one measures exactly what the probability information is
    worth. Builds in Theta(n): each key becomes a node exactly once, with no
    comparisons needed at all, since the sorted order is already known.
    """
    def build(lo: int, hi: int) -> Optional[ConventionalNode]:
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = ConventionalNode(keys[mid], probs[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(keys) - 1)


# Backwards-compatible aliases: the walkers above are generic, so the
# baseline trees use the same functions as the optimal tree.
compute_levels_conventional = compute_levels
expected_cost_conventional = verify_expected_cost
