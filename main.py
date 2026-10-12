"""
main.py

Main entry point for the Optimal Binary Search Tree (Dynamic Programming)
project.

Usage:
    python3 main.py                    -> interactive menu
    python3 main.py --example          -> runs the assignment's example input directly
    python3 main.py --experiments      -> timing experiments (n=5,10,20,50,100)
    python3 main.py --experiments-all  -> the same experiments for the uniform,
                                          random and skewed distributions
    python3 main.py --analysis         -> bonus module: brute-force optimality
                                          check, extra baselines, complexity fit
    python3 main.py --file input.txt   -> reads keys/probabilities from a file

Optional with --example / --file:
    --show-tables / --no-tables        -> force printing or hiding the DP tables
                                          (by default they are printed only for
                                          n <= MAX_TABLE_N, since a 100x100 table
                                          is unreadable in a terminal)

Input file format (see sample_input.txt):
    Line 1: n
    Next n lines: key probability
"""

import sys
import argparse

from obst_core import build_optimal_bst, reconstruct_tree
from obst_utils import (
    print_cost_table, print_root_table, print_tree,
    compute_levels, verify_expected_cost, average_depth,
    build_conventional_bst, build_degenerate_bst,
    expected_cost_conventional,
)
from experiments import run_experiments, print_experiment_table, run_all_distributions


EXAMPLE_KEYS = [10, 20, 30, 40, 50]
EXAMPLE_PROBS = [0.10, 0.20, 0.40, 0.20, 0.10]

# Above this many keys the DP tables are summarised rather than printed in full.
MAX_TABLE_N = 20


def read_input_interactive():
    """Prompts the user for n, keys, and probabilities from the console."""
    while True:
        try:
            n = int(input("Enter number of keys (n): ").strip())
            if n <= 0:
                print("n must be a positive integer. Try again.\n")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.\n")

    keys = []
    probs = []
    print(f"Enter the {n} keys in sorted order, then their search probabilities.")
    for i in range(n):
        while True:
            try:
                raw = input(f"  Key {i + 1} (value probability), e.g. '30 0.4': ").strip()
                parts = raw.split()
                if len(parts) != 2:
                    print("    Please enter exactly two values separated by a space.")
                    continue
                key = float(parts[0])
                key = int(key) if key.is_integer() else key
                prob = float(parts[1])
                keys.append(key)
                probs.append(prob)
                break
            except ValueError:
                print("    Could not parse that line. Try again.")
    return keys, probs


def read_input_from_file(path):
    """
    Reads n, keys, and probabilities from a plain text file.

    Every failure mode a malformed file can produce -- missing file, an
    unreadable count on line 1, too few key lines, a line that is not
    "<key> <probability>", or a non-numeric field -- is turned into a
    ValueError carrying a message that names the offending line, rather
    than surfacing as a raw traceback. Value-level validation (sorted
    keys, non-negative probabilities, sum ~ 1) happens later in
    obst_core.validate_input.
    """
    try:
        with open(path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        raise ValueError(f"Input file not found: {path}")
    except OSError as e:
        raise ValueError(f"Could not read input file {path}: {e}")

    if not lines:
        raise ValueError(f"Input file {path} is empty.")

    try:
        n = int(lines[0])
    except ValueError:
        raise ValueError(
            f"Line 1 of {path} must be the number of keys n (got {lines[0]!r})."
        )
    if n <= 0:
        raise ValueError(f"Number of keys must be a positive integer (got {n}).")

    data_lines = lines[1:1 + n]
    if len(data_lines) < n:
        raise ValueError(
            f"Header says {n} keys but the file only contains {len(data_lines)} "
            f"key line(s) after it."
        )

    keys, probs = [], []
    for offset, line in enumerate(data_lines):
        lineno = offset + 2  # line 1 is the count
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(
                f"Line {lineno} of {path} must be '<key> <probability>' "
                f"(got {line!r})."
            )
        try:
            key = float(parts[0])
            prob = float(parts[1])
        except ValueError:
            raise ValueError(
                f"Line {lineno} of {path} contains a non-numeric value: {line!r}."
            )
        keys.append(int(key) if key.is_integer() else key)
        probs.append(prob)

    return keys, probs


def run_full_report(keys, probs, show_tables=None):
    """
    Runs the OBST algorithm on the given input and prints every piece of
    output the assignment requires: input echo, DP tables, min cost, tree
    structure, levels, cost verification, and the conventional-BST
    comparison.

    show_tables:
        None  -> print the DP tables only when n <= MAX_TABLE_N (default);
                 a full 100 x 100 table is unreadable in a terminal and
                 buries the results underneath it.
        True  -> always print them.
        False -> never print them.
    The same threshold governs the per-key input echo and level listing.
    """
    n = len(keys)
    print("\n================ INPUT ================")
    if n <= MAX_TABLE_N:
        print(f"{'Key':>10} | {'Probability':>12}")
        for k, p in zip(keys, probs):
            print(f"{k!s:>10} | {p:>12.4f}")
    else:
        print(f"{n} keys, from {keys[0]} to {keys[-1]} "
              f"(individual keys not listed at this size).")
        print(f"Probability range: {min(probs):.6f} .. {max(probs):.6f}")
    print(f"Sum of probabilities: {sum(probs):.6f}")

    try:
        result = build_optimal_bst(keys, probs)
    except ValueError as e:
        print(f"\n[INPUT ERROR] {e}")
        return False

    if show_tables is None:
        show_tables = n <= MAX_TABLE_N

    print("\n================ DP TABLES ================")
    if show_tables:
        print_cost_table(result)
        print_root_table(result)
    else:
        print(f"Tables suppressed: n = {n} exceeds the readable threshold of "
              f"{MAX_TABLE_N}.")
        print(f"Both tables were still computed in full ({(n + 2) ** 2} cells each).")
        print("Re-run with --show-tables to print them anyway.")

    tree = reconstruct_tree(result)
    print("\n================ OPTIMAL BST ================")
    if show_tables:
        print_tree(tree)
    else:
        print(f"Tree structure suppressed at n = {n} "
              f"(re-run with --show-tables to print it).")

    levels = compute_levels(tree)
    verified_cost = verify_expected_cost(tree)

    print("\n================ RESULTS ================")
    print(f"Minimum expected search cost (from DP table C[1,n]): {result.min_cost:.6f}")
    print(f"Verified expected cost (computed by walking the tree): {verified_cost:.6f}")
    print(f"DP table build time: {result.build_time_seconds * 1000:.4f} ms")
    if show_tables:
        print("Key levels (depth, root = 1):")
        for k in keys:
            print(f"  {k}: level {levels[k]}")
    else:
        print(f"Key levels: max depth {max(levels.values())}, "
              f"average depth {sum(levels.values()) / len(levels):.4f} "
              f"(per-key listing suppressed at n = {n}).")

    print("\n================ CONVENTIONAL BST COMPARISON ================")
    deg_tree, deg_comparisons = build_degenerate_bst(keys, probs)
    conv_tree = build_conventional_bst(keys, probs)
    deg_cost = verify_expected_cost(deg_tree)
    conv_cost = expected_cost_conventional(conv_tree)

    if show_tables:
        print("(a) Degenerate BST -- keys inserted sequentially in sorted order,")
        print("    which is what a plain BST insertion routine produces here:")
        print_tree(deg_tree)
        print("\n(b) Balanced BST -- median of each range as its root,")
        print("    structurally optimal but probability-agnostic:")
        print_tree(conv_tree)
    else:
        print("Both baseline trees built; structures suppressed at this size.")

    avg_depth_deg = average_depth(deg_tree)
    avg_depth_conv = average_depth(conv_tree)
    avg_depth_opt = average_depth(tree)

    print(f"\n{'Tree':<26} | {'Exp. cost':>10} | {'Avg depth':>10} | {'Build cost':>14}")
    print("-" * 70)
    print(f"{'Degenerate (insertion)':<26} | {deg_cost:>10.6f} | {avg_depth_deg:>10.4f} | "
          f"{deg_comparisons:>8} cmp")
    print(f"{'Balanced (median root)':<26} | {conv_cost:>10.6f} | {avg_depth_conv:>10.4f} | "
          f"{'Theta(n)':>14}")
    print(f"{'Optimal (DP)':<26} | {result.min_cost:>10.6f} | {avg_depth_opt:>10.4f} | "
          f"{result.operation_count:>8} ops")

    print()
    if deg_cost > result.min_cost:
        pct = ((deg_cost - result.min_cost) / deg_cost) * 100
        print(f"-> Optimal BST beats the degenerate BST by {deg_cost - result.min_cost:.6f} "
              f"({pct:.2f}%).")
    if conv_cost > result.min_cost:
        savings = conv_cost - result.min_cost
        pct = (savings / conv_cost) * 100
        print(f"-> Optimal BST beats the balanced BST by {savings:.6f} ({pct:.2f}%).")
        print("   This second gap is the one that matters: the balanced tree is already")
        print("   structurally optimal, so the difference is due purely to using the")
        print("   search probabilities.")
    elif abs(conv_cost - result.min_cost) < 1e-9:
        print("-> The balanced tree happens to achieve the optimal cost for this input,")
        print("   which means the probabilities are flat enough that structure alone")
        print("   is already the best you can do.")
    else:
        print("-> Unexpected: a baseline beat optimal (this should not happen; check inputs).")

    return True


def run_analysis():
    """
    Runs the optional analysis module (obst_analysis.py). Everything here is
    beyond the assignment's requirements -- it is kept reachable from the
    menu and the --analysis flag so the extra work is not buried in a file
    nobody runs.
    """
    from obst_analysis import (
        check_dp_against_brute_force, compare_all_strategies,
        empirical_complexity_fit,
    )
    from experiments import (generate_random_input, generate_uniform_input,
                             generate_skewed_input)

    check_dp_against_brute_force(max_n=8)

    print("\n\n=== Strategy comparison across distributions ===")
    for gen, name in ((generate_uniform_input, "Uniform"),
                      (generate_random_input, "Random"),
                      (generate_skewed_input, "Skewed (hot key)")):
        keys, probs = gen(100)
        compare_all_strategies(keys, probs, label=name)

    # Required algorithm: confirms the Theta(n^3) growth rate empirically.
    empirical_complexity_fit(sizes=(25, 50, 100, 200), method="classic")

    # Bonus only: the same fit for the optional Knuth refinement.
    empirical_complexity_fit(method="knuth")


def main():
    parser = argparse.ArgumentParser(description="Optimal BST via Dynamic Programming")
    parser.add_argument("--example", action="store_true", help="Run the assignment's example input")
    parser.add_argument("--experiments", action="store_true", help="Run timing experiments (n=5,10,20,50,100)")
    parser.add_argument("--experiments-all", action="store_true",
                        help="Run the timing experiments for the uniform, random and skewed distributions")
    parser.add_argument("--analysis", action="store_true",
                        help="Bonus module: brute-force optimality check, extra baselines, complexity fit")
    parser.add_argument("--file", type=str, help="Read input from a file (see sample_input.txt format)")

    tables = parser.add_mutually_exclusive_group()
    tables.add_argument("--show-tables", dest="show_tables", action="store_true", default=None,
                        help=f"Always print the DP tables and tree (default: only when n <= {MAX_TABLE_N})")
    tables.add_argument("--no-tables", dest="show_tables", action="store_false",
                        help="Never print the DP tables or tree")
    args = parser.parse_args()

    if args.experiments:
        results = run_experiments()
        print_experiment_table(results)
        return

    if args.experiments_all:
        run_all_distributions()
        return

    if args.analysis:
        run_analysis()
        return

    if args.example:
        if not run_full_report(EXAMPLE_KEYS, EXAMPLE_PROBS, show_tables=args.show_tables):
            sys.exit(1)
        return

    if args.file:
        try:
            keys, probs = read_input_from_file(args.file)
        except ValueError as e:
            print(f"[INPUT ERROR] {e}")
            sys.exit(1)
        if not run_full_report(keys, probs, show_tables=args.show_tables):
            sys.exit(1)
        return

    # Interactive menu
    print("Optimal Binary Search Tree - Dynamic Programming")
    print("1. Run the assignment's example input")
    print("2. Enter my own input")
    print("3. Run timing experiments (n = 5, 10, 20, 50, 100)")
    print("4. Load input from a file")
    print("5. Run experiments across all distributions (uniform, random, skewed)")
    print("6. Run the bonus analysis module (brute-force check, extra baselines,")
    print("   empirical complexity fit)")
    choice = input("Choose an option (1-6): ").strip()

    if choice == "1":
        run_full_report(EXAMPLE_KEYS, EXAMPLE_PROBS)
    elif choice == "2":
        keys, probs = read_input_interactive()
        run_full_report(keys, probs)
    elif choice == "3":
        results = run_experiments()
        print_experiment_table(results)
    elif choice == "4":
        path = input("Enter file path: ").strip()
        try:
            keys, probs = read_input_from_file(path)
        except ValueError as e:
            print(f"[INPUT ERROR] {e}")
            return
        run_full_report(keys, probs)
    elif choice == "5":
        run_all_distributions()
    elif choice == "6":
        run_analysis()
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
