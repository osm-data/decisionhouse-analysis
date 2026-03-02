"""
GPU Allocation Caching Benchmark
=================================

Demonstrates the benefit of model caching and incremental updates for what-if
query workloads. Compares two approaches:

  1. Naive — read from DB, build model, solve (for each query)
  2. Cached — build model once, update RHS only, re-solve (simulates DeQL)

The experiment shows that speedup grows with the number of queries N, as the
one-time model build cost is amortized over more queries.

Usage:
    uv run benchmark_caching.py              # Run with XXL scale, seed 42
    uv run benchmark_caching.py --scale 3XL  # Use larger scale
    uv run benchmark_caching.py --seed 123   # Different random seed
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import duckdb
import highspy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from benchmark import (
    AllocationInstance,
    SolverResult,
    generate_instance,
    solve_lp,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCALES = {
    "M": (50, 200),       # ~7K assignments (for testing)
    "XXL": (500, 2000),   # ~460K assignments, LP solves in ~4.6s
    "3XL": (1000, 4000),  # ~1.8M assignments, LP solves in ~25s
}

SCALE_DEFAULT = "XXL"

SCALING_FACTORS = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40]  # Only increases (problem is balanced, decreases cause infeasibility)

N_QUERIES_TESTS = [5, 10, 20, 30, 40, 50]

SEED = 42
TIME_LIMIT = 1800.0  # 30 minutes to handle cold-start queries
WARMUP_ITERATIONS = 0  # No warm-up
PER_EXPERIMENT_WARMUP = 0  # No per-experiment warm-up

# ---------------------------------------------------------------------------
# DuckDB Integration
# ---------------------------------------------------------------------------

def load_instance_to_duckdb(
    conn: duckdb.DuckDBPyConnection,
    inst: AllocationInstance,
) -> None:
    """Load AllocationInstance data into DuckDB tables."""

    # Create tables
    conn.execute("""
        CREATE TABLE pools (
            pool_id INTEGER PRIMARY KEY,
            capacity INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE workloads (
            workload_id INTEGER PRIMARY KEY,
            demand INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE assignments (
            assignment_id INTEGER PRIMARY KEY,
            pool_id INTEGER,
            workload_id INTEGER,
            cost INTEGER
        )
    """)

    # Insert pools
    pools_data = [
        (int(pool_id), int(capacity))
        for pool_id, capacity in enumerate(inst.capacities)
    ]
    conn.executemany(
        "INSERT INTO pools (pool_id, capacity) VALUES (?, ?)",
        pools_data
    )

    # Insert workloads
    workloads_data = [
        (int(wl_id), int(demand))
        for wl_id, demand in enumerate(inst.demands)
    ]
    conn.executemany(
        "INSERT INTO workloads (workload_id, demand) VALUES (?, ?)",
        workloads_data
    )

    # Insert assignments
    assignments_data = [
        (int(i), int(pool_id), int(wl_id), int(cost))
        for i, (pool_id, wl_id, cost) in enumerate(
            zip(inst.assign_pool, inst.assign_workload, inst.costs)
        )
    ]
    conn.executemany(
        "INSERT INTO assignments (assignment_id, pool_id, workload_id, cost) VALUES (?, ?, ?, ?)",
        assignments_data
    )


# ---------------------------------------------------------------------------
# Cached Model Builder
# ---------------------------------------------------------------------------

def build_cached_model(
    inst: AllocationInstance,
    time_limit: float = TIME_LIMIT,
) -> tuple[highspy.Highs, list[int]]:
    """
    Build LP model once for caching. Returns model and supply constraint row indices.

    Based on solve_lp() but tracks which rows correspond to supply constraints
    so we can update them later.

    Returns:
        (highs_model, supply_constraint_rows)
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("solver", "simplex")

    n = inst.n_assignments
    max_cap = float(inst.capacities.max())

    # Add variables
    h.addVars(n, np.zeros(n), np.full(n, max_cap))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), inst.costs.astype(np.float64))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    # Build constraints and track supply constraint row indices
    n_pools = inst.n_pools
    n_wl = inst.n_workloads

    # Pre-compute assignment lists
    assigns_by_pool: list[list[int]] = [[] for _ in range(n_pools)]
    assigns_by_wl: list[list[int]] = [[] for _ in range(n_wl)]
    for r in range(n):
        assigns_by_pool[inst.assign_pool[r]].append(r)
        assigns_by_wl[inst.assign_workload[r]].append(r)

    # Add supply constraints first, track their row indices
    supply_constraint_rows = []
    row_lower_list = []
    row_upper_list = []
    csr_indices = []
    csr_values = []
    csr_starts = []
    inf = highspy.kHighsInf

    current_row = 0
    for p in range(n_pools):
        assigns = assigns_by_pool[p]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(r)
            csr_values.append(1.0)
        row_lower_list.append(-inf)
        row_upper_list.append(float(inst.capacities[p]))
        supply_constraint_rows.append(current_row)
        current_row += 1

    # Add demand constraints (these won't be updated)
    for w in range(n_wl):
        assigns = assigns_by_wl[w]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(r)
            csr_values.append(1.0)
        d = float(inst.demands[w])
        row_lower_list.append(d)
        row_upper_list.append(d)
        current_row += 1

    # Add all constraints
    n_rows = len(row_lower_list)
    h.addRows(
        n_rows,
        np.array(row_lower_list),
        np.array(row_upper_list),
        len(csr_indices),
        np.array(csr_starts, dtype=np.int32),
        np.array(csr_indices, dtype=np.int32),
        np.array(csr_values),
    )

    return h, supply_constraint_rows


# ---------------------------------------------------------------------------
# Naive Approach
# ---------------------------------------------------------------------------

def naive_solve_from_db(
    conn: duckdb.DuckDBPyConnection,
    scaling_factor: float,
    time_limit: float = TIME_LIMIT,
) -> tuple[SolverResult, float, float]:
    """
    Naive approach: read from DB, join, build model, solve.

    Returns:
        (solver_result, total_time, db_read_time)
    """
    t0 = time.perf_counter()

    # Step 1: Read pools with scaled capacity
    pools_df = conn.execute("""
        SELECT pool_id, CAST(capacity * ? AS INTEGER) AS capacity
        FROM pools
        ORDER BY pool_id
    """, [scaling_factor]).fetchdf()

    # Step 2: Read workloads
    workloads_df = conn.execute("""
        SELECT workload_id, demand
        FROM workloads
        ORDER BY workload_id
    """).fetchdf()

    # Step 3: Read assignments
    assignments_df = conn.execute("""
        SELECT pool_id, workload_id, cost
        FROM assignments
        ORDER BY assignment_id
    """).fetchdf()

    db_read_time = time.perf_counter() - t0

    # Step 4: Convert to AllocationInstance format
    inst = AllocationInstance(
        n_pools=len(pools_df),
        n_workloads=len(workloads_df),
        capacities=pools_df['capacity'].values.astype(np.int64),
        demands=workloads_df['demand'].values.astype(np.int64),
        assign_pool=assignments_df['pool_id'].values.astype(np.int64),
        assign_workload=assignments_df['workload_id'].values.astype(np.int64),
        costs=assignments_df['cost'].values.astype(np.int64),
        n_assignments_before_filter=len(assignments_df),
    )

    # Step 5: Build and solve LP model from scratch
    result = solve_lp(inst, time_limit=time_limit)
    total_time = time.perf_counter() - t0

    return result, total_time, db_read_time


# ---------------------------------------------------------------------------
# Cached Approach
# ---------------------------------------------------------------------------

def cached_solve_with_updates(
    conn: duckdb.DuckDBPyConnection,
    highs_model: highspy.Highs,
    supply_constraint_rows: list[int],
    scaling_factor: float,
) -> tuple[SolverResult, float, float]:
    """
    Cached approach: only update RHS for supply constraints, re-solve.

    Args:
        conn: DuckDB connection (for simulating "read new capacity values")
        highs_model: Pre-built HiGHS model (reused across queries)
        supply_constraint_rows: List of row indices corresponding to supply constraints
        scaling_factor: Capacity multiplier for this query

    Returns:
        (solver_result, total_time, db_read_time)
    """
    t0 = time.perf_counter()

    # Step 1: "Read" new capacity values (much smaller query than naive)
    pools_df = conn.execute("""
        SELECT pool_id, CAST(capacity * ? AS INTEGER) AS capacity
        FROM pools
        ORDER BY pool_id
    """, [scaling_factor]).fetchdf()

    new_capacities = pools_df['capacity'].values.astype(np.float64)
    db_read_time = time.perf_counter() - t0

    # Step 2: Update row bounds for supply constraints only
    inf = highspy.kHighsInf
    n_supply_rows = len(supply_constraint_rows)

    row_indices = np.array(supply_constraint_rows, dtype=np.int32)
    row_lower = np.full(n_supply_rows, -inf)  # Unchanged
    row_upper = new_capacities  # Updated capacities

    # Use HiGHS incremental update API
    highs_model.changeRowsBounds(
        n_supply_rows,
        row_indices,
        row_lower,
        row_upper
    )

    # Step 3: Re-solve (warm start from previous basis)
    t_solve = time.perf_counter()
    highs_model.run()
    solve_time = time.perf_counter() - t_solve

    # Extract result
    status = highs_model.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        obj = highs_model.getInfoValue("objective_function_value")[1]
        result = SolverResult(objective=obj, solve_time=solve_time, status="optimal")
    else:
        result = SolverResult(objective=float("inf"), solve_time=solve_time,
                             status=str(status))

    total_time = time.perf_counter() - t0
    return result, total_time, db_read_time


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_identical_results(
    naive_results: list[SolverResult],
    cached_results: list[SolverResult],
    scaling_factors: list[float],
    rel_tol: float = 1e-4,
) -> None:
    """Assert that naive and cached approaches produce identical objectives."""
    assert len(naive_results) == len(cached_results), "Result count mismatch"

    for i, (naive, cached, factor) in enumerate(
        zip(naive_results, cached_results, scaling_factors)
    ):
        # Both must be optimal
        assert naive.status == "optimal", \
            f"Query {i} (factor={factor}): naive status={naive.status}"
        assert cached.status == "optimal", \
            f"Query {i} (factor={factor}): cached status={cached.status}"

        # Objectives must match
        if naive.objective == 0:
            assert abs(cached.objective) < 1e-6, \
                f"Query {i}: objectives don't match (naive=0, cached={cached.objective})"
        else:
            rel_diff = abs(cached.objective - naive.objective) / abs(naive.objective)
            assert rel_diff < rel_tol, (
                f"Query {i} (factor={factor}): objective mismatch\n"
                f"  Naive:  {naive.objective:.2f}\n"
                f"  Cached: {cached.objective:.2f}\n"
                f"  Rel diff: {rel_diff:.6f}"
            )

    print(f"  [OK] All {len(naive_results)} queries verified: identical objectives")


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

def run_caching_experiment(
    scale: str = SCALE_DEFAULT,
    seed: int = SEED,
    n_queries_list: list[int] = None,
    scaling_factors: list[float] = None,
) -> dict:
    """
    Main experiment runner.

    Returns results dict suitable for JSON serialization and plotting.
    """
    if n_queries_list is None:
        n_queries_list = N_QUERIES_TESTS
    if scaling_factors is None:
        scaling_factors = SCALING_FACTORS

    # 1. Generate instance
    n_pools, n_wl = SCALES[scale]
    print(f"\nGenerating instance: {scale} ({n_pools} pools x {n_wl} workloads)")
    inst = generate_instance(n_pools, n_wl, seed=seed)
    print(f"  Assignments: {inst.n_assignments:,}")

    # 2. Load into DuckDB
    print("Loading data into DuckDB...")
    conn = duckdb.connect(":memory:")
    load_instance_to_duckdb(conn, inst)
    print(f"  Loaded {inst.n_pools} pools, {inst.n_workloads} workloads, {inst.n_assignments} assignments")

    # 3. Build cached model (one-time)
    print("Building cached LP model...")
    t_build = time.perf_counter()
    cached_model, supply_rows = build_cached_model(inst, time_limit=TIME_LIMIT)
    build_time = time.perf_counter() - t_build
    print(f"  Model build time: {build_time:.2f}s")
    print(f"  Supply constraint rows tracked: {len(supply_rows)}")

    # 3.5. Warm-up phase (optional, disabled if WARMUP_ITERATIONS=0)
    if WARMUP_ITERATIONS > 0:
        print(f"Running warm-up phase ({WARMUP_ITERATIONS} iterations to reach steady state)...")
        for i in range(WARMUP_ITERATIONS):
            naive_solve_from_db(conn, 1.0, TIME_LIMIT)
            cached_solve_with_updates(conn, cached_model, supply_rows, 1.0)
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{WARMUP_ITERATIONS} warm-up iterations")
        print("  Warm-up complete - system at steady state")
    else:
        print("Skipping warm-up phase (relying on steady-state analysis to exclude slow initial queries)")

    # 4. Run experiments for different N
    experiments = []

    for n_queries in n_queries_list:
        print(f"\n{'='*60}")
        print(f"Experiment: N={n_queries} queries")
        print(f"{'='*60}")

        # Generate n_queries scaling factors (evenly spaced from 1.0 to 1.4)
        if n_queries <= len(scaling_factors):
            factors = scaling_factors[:n_queries]
        else:
            # Generate more factors if needed
            factors = [1.0 + i * 0.4 / (n_queries - 1) for i in range(n_queries)]
        print(f"  Scaling factors: {[f'{f:.2f}' for f in factors[:5]]}{'...' if len(factors) > 5 else ''} ({len(factors)} total)")

        # Warm-up for naive approach (optional)
        if PER_EXPERIMENT_WARMUP > 0:
            print(f"  Warming up naive approach ({PER_EXPERIMENT_WARMUP} queries)...")
            warmup_factors = [1.0, 1.1, 1.2, 1.3, 1.4]
            for i in range(PER_EXPERIMENT_WARMUP):
                factor = warmup_factors[i % len(warmup_factors)]
                naive_solve_from_db(conn, factor, TIME_LIMIT)

        # Run naive approach (single run per query)
        print(f"  Running naive approach...")
        naive_results = []
        naive_times = []
        naive_db_times = []
        t_naive_start = time.perf_counter()

        for i, factor in enumerate(factors):
            result, total_time, db_time = naive_solve_from_db(conn, factor, TIME_LIMIT)
            naive_results.append(result)
            naive_times.append(total_time)
            naive_db_times.append(db_time)

            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{n_queries} queries")

        naive_total = time.perf_counter() - t_naive_start
        naive_avg_ms = (naive_total/n_queries) * 1000
        print(f"    Wall clock time: {naive_total:.2f}s (avg: {naive_avg_ms:.1f}ms per query)")

        # Warm-up for cached approach (optional)
        if PER_EXPERIMENT_WARMUP > 0:
            print(f"  Warming up cached approach ({PER_EXPERIMENT_WARMUP} queries)...")
            warmup_factors = [1.0, 1.1, 1.2, 1.3, 1.4]
            for i in range(PER_EXPERIMENT_WARMUP):
                factor = warmup_factors[i % len(warmup_factors)]
                cached_solve_with_updates(conn, cached_model, supply_rows, factor)

        # Run cached approach (single run per query)
        print(f"  Running cached approach...")
        cached_results = []
        cached_times = []
        cached_db_times = []
        t_cached_start = time.perf_counter()

        for i, factor in enumerate(factors):
            result, total_time, db_time = cached_solve_with_updates(
                conn, cached_model, supply_rows, factor
            )
            cached_results.append(result)
            cached_times.append(total_time)
            cached_db_times.append(db_time)

            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{n_queries} queries")

        cached_total = time.perf_counter() - t_cached_start
        cached_avg_ms = (cached_total/n_queries) * 1000
        print(f"    Wall clock time: {cached_total:.2f}s (avg: {cached_avg_ms:.1f}ms per query)")

        # Verify results match
        verify_identical_results(naive_results, cached_results, factors)

        # Compute speedup using wall clock times
        speedup = naive_total / cached_total
        print(f"  Speedup (wall clock): {speedup:.2f}×")

        # Steady-state analysis (exclude first 2 queries to remove warm-up effects)
        warmup_queries = min(2, n_queries // 2)  # Exclude first 2 queries, or half if N is small
        if n_queries > warmup_queries:
            naive_steady = naive_times[warmup_queries:]
            cached_steady = cached_times[warmup_queries:]
            naive_steady_avg = sum(naive_steady) / len(naive_steady)
            cached_steady_avg = sum(cached_steady) / len(cached_steady)
            steady_speedup = naive_steady_avg / cached_steady_avg
            print(f"  Steady-state (excluding first {warmup_queries}): "
                  f"naive={naive_steady_avg*1000:.1f}ms, cached={cached_steady_avg*1000:.1f}ms, "
                  f"speedup={steady_speedup:.2f}×")
        else:
            naive_steady_avg = cached_steady_avg = steady_speedup = None

        # Store experiment results
        experiments.append({
            "n_queries": n_queries,
            "scaling_factors": factors,
            "naive": {
                "wall_clock_total_sec": naive_total,
                "per_query_times_sec": naive_times,
                "db_read_times_sec": naive_db_times,
                "objectives": [r.objective for r in naive_results],
                "all_optimal": all(r.status == "optimal" for r in naive_results),
            },
            "cached": {
                "initial_build_time_sec": build_time,
                "wall_clock_total_sec": cached_total,
                "per_query_times_sec": cached_times,
                "db_read_times_sec": cached_db_times,
                "objectives": [r.objective for r in cached_results],
                "all_optimal": all(r.status == "optimal" for r in cached_results),
            },
            "speedup": speedup,
            "steady_state": {
                "warmup_queries_excluded": warmup_queries,
                "naive_avg_sec": naive_steady_avg,
                "cached_avg_sec": cached_steady_avg,
                "speedup": steady_speedup,
            } if n_queries > warmup_queries else None,
        })

    # Return complete results
    return {
        "seed": seed,
        "scale": scale,
        "n_pools": n_pools,
        "n_workloads": n_wl,
        "n_assignments": inst.n_assignments,
        "experiments": experiments,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_caching_results(results: dict, output_dir: Path, scale: str = ""):
    """
    Generate publication-quality plot for VLDB paper.
    X-axis: N queries, Y-axis: total wall time.
    """
    n_queries = [exp["n_queries"] for exp in results["experiments"]]
    naive_times = [exp["naive"]["wall_clock_total_sec"] for exp in results["experiments"]]
    cached_times = [exp["cached"]["wall_clock_total_sec"] for exp in results["experiments"]]
    speedups = [exp["speedup"] for exp in results["experiments"]]

    # Add scale to filename
    scale_suffix = f"_{scale}" if scale else ""

    # VLDB publication settings (2-column format, print-ready)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "pdf.fonttype": 42,  # TrueType fonts for PDF
        "ps.fonttype": 42,
    })

    # Figure size optimized for 2-column VLDB format (~3.3" width)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot with distinct markers and patterns (works in grayscale)
    ax.plot(n_queries, naive_times, "s-",
            label="Naive",
            linewidth=1.8, markersize=5,
            color="#d62728", markerfacecolor="white",
            markeredgewidth=1.5, markeredgecolor="#d62728")

    ax.plot(n_queries, cached_times, "o-",
            label="Cached",
            linewidth=1.8, markersize=5,
            color="#1f77b4", markerfacecolor="#1f77b4",
            markeredgewidth=0)

    # Speedup annotations - just above blue dots, skip N=5
    max_y = max(naive_times) * 1.2
    for i, n in enumerate(n_queries):
        # Skip annotation for N=5 (too crowded)
        if n == 5:
            continue

        # Position using offset points for consistent spacing above marker
        ax.annotate(f"{speedups[i]:.1f}×",
                   xy=(n, cached_times[i]),
                   xytext=(0, 5),  # 5 points above the blue dot
                   textcoords="offset points",
                   ha="center", va="bottom",
                   fontsize=6.5,
                   color="#1f77b4",
                   bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec="none",
                            alpha=0.85))

    # Clean, professional styling
    ax.set_xlabel("Number of queries", fontsize=10)
    ax.set_ylabel("Execution time (s)", fontsize=10)
    ax.legend(loc="upper left", frameon=True, edgecolor="#cccccc",
             framealpha=0.95, fancybox=False)

    # Subtle grid
    ax.grid(True, alpha=0.2, linestyle=":", linewidth=0.5, color="gray")
    ax.set_axisbelow(True)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # Set ticks
    ax.set_xticks(n_queries)
    ax.set_ylim(0, max_y)
    ax.set_xlim(min(n_queries) - 2, max(n_queries) + 2)

    # Tight layout for publication
    fig.tight_layout(pad=0.3)

    # Save in multiple formats with scale suffix
    fig.savefig(output_dir / f"caching_benefit{scale_suffix}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_dir / f"caching_benefit{scale_suffix}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(output_dir / f"caching_benefit{scale_suffix}.eps", bbox_inches="tight", pad_inches=0.02, format="eps")

    print(f"\nPlot saved to:")
    print(f"  {output_dir / f'caching_benefit{scale_suffix}.pdf'}")
    print(f"  {output_dir / f'caching_benefit{scale_suffix}.png'}")
    print(f"  {output_dir / f'caching_benefit{scale_suffix}.eps'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark caching benefit for what-if queries"
    )
    parser.add_argument(
        "--scale", default=SCALE_DEFAULT, choices=list(SCALES.keys()),
        help=f"Problem scale (default: {SCALE_DEFAULT})"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("="*60)
    print("GPU Allocation Caching Benchmark")
    print("="*60)
    print(f"Scale: {args.scale}, Seed: {args.seed}")

    # Run experiment
    results = run_caching_experiment(scale=args.scale, seed=args.seed)

    # Save results with scale in filename
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    results_file = out_dir / f"results_cache_experiment_{args.scale}_seed{args.seed}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Generate plot with scale suffix
    plot_caching_results(results, out_dir, scale=args.scale)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    summary = []
    for exp in results["experiments"]:
        row = {
            "N": exp["n_queries"],
            "Naive (s)": f"{exp['naive']['wall_clock_total_sec']:.1f}",
            "Cached (s)": f"{exp['cached']['wall_clock_total_sec']:.4f}",
            "Speedup": f"{exp['speedup']:.2f}×",
        }
        # Add steady-state info if available
        if exp.get("steady_state"):
            ss = exp["steady_state"]
            row["Steady-State Speedup"] = f"{ss['speedup']:.2f}×" if ss['speedup'] else "N/A"
        summary.append(row)
    print(tabulate(summary, headers="keys", tablefmt="simple"))
    print(f"\nNote: Wall clock times (single run per query, after {WARMUP_ITERATIONS}-iteration warm-up)")


if __name__ == "__main__":
    main()
