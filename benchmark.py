"""
GPU Allocation Formulation Benchmark
=====================================

Solves a GPU-to-workload allocation problem (matching the DeQL vision paper
running example) using three formulations at increasing scale:

  1. Generic MILP  — binary active + continuous quantity + big-M  (HiGHS branch-and-bound)
  2. LP relaxation  — continuous only, exploiting total unimodularity  (HiGHS simplex)
  3. Min-cost flow  — network simplex on bipartite supply-demand graph  (OR-Tools)

The problem: an AI inference platform allocates GPU compute (measured in
GPU-hours) from GPU pools to inference workloads, minimizing total cost
subject to pool capacity and workload demand constraints.

Usage:
    uv run benchmark.py              # run all scales
    uv run benchmark.py --scales S M # run specific scales
    uv run benchmark.py --seed 123   # different random seed
"""

from __future__ import annotations

import argparse
import contextlib
import json
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import highspy
import numpy as np
from ortools.graph.python import min_cost_flow
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class AllocationInstance:
    n_pools: int
    n_workloads: int
    capacities: np.ndarray      # int64, shape (n_pools,)
    demands: np.ndarray         # int64, shape (n_workloads,)
    assign_pool: np.ndarray     # int64, index into pools
    assign_workload: np.ndarray # int64, index into workloads
    costs: np.ndarray           # int64, per-assignment unit cost
    n_assignments_before_filter: int

    @property
    def n_assignments(self) -> int:
        return len(self.costs)


def generate_instance(
    n_pools: int,
    n_workloads: int,
    seed: int = 42,
    density: float = 0.7,
    latency_filter_ms: int = 200,
) -> AllocationInstance:
    """Generate a random balanced GPU allocation problem.

    GPU pools have capacities (GPU-hours), workloads have demands (GPU-hours),
    and each possible pool-workload assignment has a cost and latency.
    The WHERE latency_ms <= 200 filter removes assignments that violate SLA.

    All values are integers (required by OR-Tools SimpleMinCostFlow).
    Supply is balanced: sum(capacities) == sum(demands).
    """
    rng = np.random.default_rng(seed)

    # Raw capacities and demands (GPU-hours)
    raw_caps = rng.integers(100, 1000, size=n_pools)
    raw_demands = rng.integers(20, 200, size=n_workloads)

    # Balance: scale demands so total demand == total supply
    total_supply = int(raw_caps.sum())
    scaled = raw_demands / raw_demands.sum() * total_supply
    demands = np.maximum(scaled.astype(np.int64), 1)
    demands[-1] += total_supply - int(demands.sum())  # fix rounding
    capacities = raw_caps.astype(np.int64)

    assert int(capacities.sum()) == int(demands.sum()), "Supply-demand imbalance"

    # Generate assignments: each (pool, workload) pair exists with probability `density`
    all_pool = []
    all_workload = []
    all_cost = []
    all_latency = []
    for p in range(n_pools):
        for w in range(n_workloads):
            if rng.random() < density:
                all_pool.append(p)
                all_workload.append(w)
                all_cost.append(int(rng.integers(1, 101)))
                all_latency.append(int(rng.integers(10, 301)))  # 10..300 ms

    n_before = len(all_pool)

    # Apply WHERE latency_ms <= 200 filter (matching DeQL query)
    mask = [lat <= latency_filter_ms for lat in all_latency]
    assign_pool = np.array([all_pool[i] for i in range(n_before) if mask[i]], dtype=np.int64)
    assign_workload = np.array([all_workload[i] for i in range(n_before) if mask[i]], dtype=np.int64)
    costs = np.array([all_cost[i] for i in range(n_before) if mask[i]], dtype=np.int64)

    # Feasibility check: every workload must be reachable by at least one assignment
    reachable_workloads = set(assign_workload.tolist())
    for w in range(n_workloads):
        if w not in reachable_workloads:
            p = int(np.argmax(capacities))
            assign_pool = np.append(assign_pool, p)
            assign_workload = np.append(assign_workload, w)
            costs = np.append(costs, int(rng.integers(1, 101)))

    # Every pool must also have at least one outgoing assignment
    reachable_pools = set(assign_pool.tolist())
    for p in range(n_pools):
        if p not in reachable_pools:
            w = int(np.argmax(demands))
            assign_pool = np.append(assign_pool, p)
            assign_workload = np.append(assign_workload, w)
            costs = np.append(costs, int(rng.integers(1, 101)))

    return AllocationInstance(
        n_pools=n_pools,
        n_workloads=n_workloads,
        capacities=capacities,
        demands=demands,
        assign_pool=assign_pool,
        assign_workload=assign_workload,
        costs=costs,
        n_assignments_before_filter=n_before,
    )


# ---------------------------------------------------------------------------
# Memory limit helper
# ---------------------------------------------------------------------------

MEM_LIMIT_GB = 14  # cap solver memory to avoid OOM-killing the whole process

@contextlib.contextmanager
def memory_limit(gb: float = MEM_LIMIT_GB):
    """Temporarily set RLIMIT_AS so a MemoryError is raised instead of OOM kill."""
    limit_bytes = int(gb * 1024**3)
    soft_old, hard_old = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, hard_old))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft_old, hard_old))


# ---------------------------------------------------------------------------
# Formulation 1: Generic MILP (binary active + continuous quantity + big-M)
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    objective: float
    solve_time: float  # seconds
    status: str

def _build_constraint_csr(inst: AllocationInstance, qty_offset: int = 0):
    """Build supply + demand constraint rows in CSR format.

    Returns (n_rows, row_lower, row_upper, nnz, starts, indices, values).
    qty_offset: column offset where quantity variables start (0 for LP, n for MILP).
    """
    n = inst.n_assignments
    n_pools = inst.n_pools
    n_wl = inst.n_workloads

    # Pre-compute assignment lists per pool and workload
    assigns_by_pool: list[list[int]] = [[] for _ in range(n_pools)]
    assigns_by_wl: list[list[int]] = [[] for _ in range(n_wl)]
    for r in range(n):
        assigns_by_pool[inst.assign_pool[r]].append(r)
        assigns_by_wl[inst.assign_workload[r]].append(r)

    row_lower_list = []
    row_upper_list = []
    csr_indices = []
    csr_values = []
    csr_starts = []
    inf = highspy.kHighsInf

    # Supply constraints: SUM(quantity) BY pool_id <= capacity
    for p in range(n_pools):
        assigns = assigns_by_pool[p]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(qty_offset + r)
            csr_values.append(1.0)
        row_lower_list.append(-inf)
        row_upper_list.append(float(inst.capacities[p]))

    # Demand constraints: SUM(quantity) BY workload_id == demand
    for w in range(n_wl):
        assigns = assigns_by_wl[w]
        if not assigns:
            continue
        csr_starts.append(len(csr_indices))
        for r in assigns:
            csr_indices.append(qty_offset + r)
            csr_values.append(1.0)
        d = float(inst.demands[w])
        row_lower_list.append(d)
        row_upper_list.append(d)

    n_rows = len(row_lower_list)
    return (
        n_rows,
        np.array(row_lower_list),
        np.array(row_upper_list),
        len(csr_indices),
        np.array(csr_starts, dtype=np.int32),
        np.array(csr_indices, dtype=np.int32),
        np.array(csr_values),
    )


def solve_milp(inst: AllocationInstance, time_limit: float = 600.0) -> SolverResult:
    """MILP formulation: what the user writes (no structure recognition).

    Uses batch model construction for performance at large scale.
    Variables: active[0..n-1] binary, quantity[n..2n-1] continuous.
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    n = inst.n_assignments
    big_m = float(inst.capacities.max())
    inf = highspy.kHighsInf

    # Add 2n variables in batch: active[0..n-1], quantity[n..2n-1]
    col_lower = np.zeros(2 * n)
    col_upper = np.concatenate([np.ones(n), np.full(n, big_m)])
    h.addVars(2 * n, col_lower, col_upper)

    # Set objective costs: 0 for active, cost[r] for quantity
    col_costs = np.concatenate([np.zeros(n), inst.costs.astype(np.float64)])
    h.changeColsCost(2 * n, np.arange(2 * n, dtype=np.int32), col_costs)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    # Mark active[0..n-1] as integer (binary)
    h.changeColsIntegrality(
        n, np.arange(n, dtype=np.int32),
        np.full(n, highspy.HighsVarType.kInteger),
    )

    # Build constraints in CSR format

    # 1. Big-M linking: quantity[r] - big_m * active[r] <= 0, for r=0..n-1
    bigm_starts = np.arange(n, dtype=np.int32) * 2
    bigm_indices = np.empty(2 * n, dtype=np.int32)
    bigm_values = np.empty(2 * n)
    for r in range(n):
        bigm_indices[2 * r] = r          # active[r]
        bigm_values[2 * r] = -big_m
        bigm_indices[2 * r + 1] = n + r  # quantity[r]
        bigm_values[2 * r + 1] = 1.0
    bigm_lower = np.full(n, -inf)
    bigm_upper = np.zeros(n)

    h.addRows(n, bigm_lower, bigm_upper, 2 * n, bigm_starts, bigm_indices, bigm_values)

    # 2. Supply + demand constraints (quantity vars start at column n)
    sd_rows, sd_lower, sd_upper, sd_nnz, sd_starts, sd_indices, sd_values = \
        _build_constraint_csr(inst, qty_offset=n)
    h.addRows(sd_rows, sd_lower, sd_upper, sd_nnz, sd_starts, sd_indices, sd_values)

    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        obj = h.getInfoValue("objective_function_value")[1]
        return SolverResult(objective=obj, solve_time=solve_time, status="optimal")
    elif status == highspy.HighsModelStatus.kTimeLimit:
        try:
            obj = h.getInfoValue("objective_function_value")[1]
        except Exception:
            obj = float("inf")
        return SolverResult(objective=obj, solve_time=solve_time, status="timeout")
    else:
        return SolverResult(objective=float("inf"), solve_time=solve_time, status=str(status))


# ---------------------------------------------------------------------------
# Formulation 2: LP relaxation (DeOP detects TU, drops binary variables)
# ---------------------------------------------------------------------------

def solve_lp(inst: AllocationInstance, time_limit: float = 600.0) -> SolverResult:
    """LP formulation: DeOP recognizes total unimodularity, binary vars are redundant.

    Uses batch model construction. Variables: quantity[0..n-1] continuous only.
    """
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)
    h.setOptionValue("solver", "simplex")

    n = inst.n_assignments
    max_cap = float(inst.capacities.max())

    # Add n continuous variables in batch
    h.addVars(n, np.zeros(n), np.full(n, max_cap))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), inst.costs.astype(np.float64))
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    # Supply + demand constraints (quantity vars start at column 0)
    sd_rows, sd_lower, sd_upper, sd_nnz, sd_starts, sd_indices, sd_values = \
        _build_constraint_csr(inst, qty_offset=0)
    h.addRows(sd_rows, sd_lower, sd_upper, sd_nnz, sd_starts, sd_indices, sd_values)

    t0 = time.perf_counter()
    h.run()
    solve_time = time.perf_counter() - t0

    status = h.getModelStatus()
    if status == highspy.HighsModelStatus.kOptimal:
        obj = h.getInfoValue("objective_function_value")[1]
        return SolverResult(objective=obj, solve_time=solve_time, status="optimal")
    elif status == highspy.HighsModelStatus.kTimeLimit:
        try:
            obj = h.getInfoValue("objective_function_value")[1]
        except Exception:
            obj = float("inf")
        return SolverResult(objective=obj, solve_time=solve_time, status="timeout")
    else:
        return SolverResult(objective=float("inf"), solve_time=solve_time, status=str(status))


# ---------------------------------------------------------------------------
# Formulation 3: Min-cost flow (DeOP detects bipartite network structure)
# ---------------------------------------------------------------------------

def solve_mcf(inst: AllocationInstance) -> SolverResult:
    """Min-cost flow formulation: DeOP recognizes bipartite supply-demand network."""
    smcf = min_cost_flow.SimpleMinCostFlow()

    n_pools = inst.n_pools
    n_wl = inst.n_workloads
    n = inst.n_assignments

    # Nodes: 0..n_pools-1 are GPU pools, n_pools..n_pools+n_wl-1 are workloads
    start_nodes = inst.assign_pool.astype(np.int64)
    end_nodes = (n_pools + inst.assign_workload).astype(np.int64)

    # Arc capacity: use pool capacity (safe upper bound per arc)
    arc_caps = inst.capacities[inst.assign_pool].astype(np.int64)
    unit_costs = inst.costs.astype(np.int64)

    smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, arc_caps, unit_costs)

    # Node supplies: positive = source (pool), negative = sink (workload)
    n_nodes = n_pools + n_wl
    supplies = np.zeros(n_nodes, dtype=np.int64)
    supplies[:n_pools] = inst.capacities
    supplies[n_pools:] = -inst.demands

    smcf.set_nodes_supplies(np.arange(n_nodes, dtype=np.int64), supplies)

    t0 = time.perf_counter()
    status = smcf.solve()
    solve_time = time.perf_counter() - t0

    if status == smcf.OPTIMAL:
        obj = float(smcf.optimal_cost())
        return SolverResult(objective=obj, solve_time=solve_time, status="optimal")
    else:
        status_names = {
            smcf.NOT_SOLVED: "not_solved",
            smcf.INFEASIBLE: "infeasible",
            smcf.UNBALANCED: "unbalanced",
            smcf.BAD_COST_RANGE: "bad_cost_range",
        }
        return SolverResult(
            objective=float("inf"),
            solve_time=solve_time,
            status=status_names.get(status, f"unknown({status})"),
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_results(
    inst: AllocationInstance,
    milp: SolverResult,
    lp: SolverResult,
    mcf: SolverResult,
    rel_tol: float = 1e-4,
) -> None:
    """Assert that all three formulations agree on optimal cost."""
    results = {"MILP": milp, "LP": lp, "MCF": mcf}

    # All must be optimal (MILP may timeout at large scale — skip if so)
    optimal_results = {k: v for k, v in results.items() if v.status == "optimal"}

    if len(optimal_results) < 2:
        print(f"  [WARN] Only {len(optimal_results)} solver(s) reached optimality, skipping cross-check")
        return

    # Cross-check: all optimal objectives must agree
    names = list(optimal_results.keys())
    objs = [optimal_results[k].objective for k in names]
    ref = objs[0]
    for name, obj in zip(names, objs):
        if ref == 0:
            assert abs(obj) < 1e-6, f"{name} objective {obj} != 0"
        else:
            rel_diff = abs(obj - ref) / abs(ref)
            assert rel_diff < rel_tol, (
                f"Objective mismatch: {names[0]}={ref:.2f}, {name}={obj:.2f} "
                f"(rel diff={rel_diff:.6f})"
            )

    print(f"  [OK] Objectives match across {', '.join(names)}: {ref:.2f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

SCALES = {
    "S":    (20, 60),
    "M":    (50, 200),
    "L":    (100, 400),
    "XL":   (200, 800),
    "XXL":  (500, 2000),
    "3XL":  (1000, 4000),
    "4XL":  (2000, 8000),
}

MILP_TIME_LIMIT = 600.0  # 10 minutes
LP_TIME_LIMIT = 600.0
N_RUNS = 3


def format_time(seconds: float, status: str) -> str:
    if status == "timeout":
        return f">{MILP_TIME_LIMIT:.0f}s"
    if seconds < 0.01:
        return f"{seconds*1000:.1f}ms"
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}min"


def format_speedup(slow: SolverResult, fast: SolverResult) -> str:
    if slow.status == "timeout":
        return f">{MILP_TIME_LIMIT / fast.solve_time:.0f}x"
    if fast.solve_time < 1e-9:
        return ">999999x"
    ratio = slow.solve_time / fast.solve_time
    if ratio >= 1000:
        return f"{ratio/1000:.1f}Kx"
    return f"{ratio:.1f}x"


def run_benchmark(scales: list[str], seed: int, n_runs: int) -> None:
    rows = []

    for label in scales:
        if label not in SCALES:
            print(f"Unknown scale: {label}")
            continue

        n_pools, n_wl = SCALES[label]
        print(f"\n{'='*60}")
        print(f"Scale {label} ({n_pools} GPU pools x {n_wl} workloads)")
        print(f"{'='*60}")

        inst = generate_instance(n_pools, n_wl, seed=seed)
        print(f"  Assignments: {inst.n_assignments} (before filter: {inst.n_assignments_before_filter})")
        print(f"  Total supply/demand: {int(inst.capacities.sum())}")

        # Run each formulation n_runs times, take median
        mcf_times = []
        lp_times = []
        milp_times = []
        milp_status = "optimal"
        lp_status = "optimal"
        mcf_status = "optimal"

        def _run_solver(name, solve_fn, times_list, n_runs, stop_on_timeout=False):
            """Run a solver n_runs times with memory limit. Returns (result, status)."""
            print(f"  Running {name} ({n_runs} runs)...", end=" ", flush=True)
            result = None
            status = "optimal"
            try:
                with memory_limit():
                    for i in range(n_runs):
                        t_start = time.perf_counter()
                        try:
                            r = solve_fn()
                        except MemoryError:
                            elapsed = time.perf_counter() - t_start
                            print(f"OOM after {elapsed:.2f}s", end=" ")
                            status = "oom"
                            result = SolverResult(float("inf"), elapsed, "oom")
                            times_list.append(elapsed)
                            break
                        times_list.append(r.solve_time)
                        result = r
                        status = r.status
                        if stop_on_timeout and r.status == "timeout":
                            print(f"timeout on run {i+1}", end=" ")
                            break
            except MemoryError:
                elapsed = time.perf_counter() - t_start
                print(f"OOM after {elapsed:.2f}s", end=" ")
                status = "oom"
                if result is None:
                    result = SolverResult(float("inf"), elapsed, "oom")
                    times_list.append(elapsed)
            if times_list:
                print(f"{statistics.median(times_list)*1000:.2f}ms median")
            else:
                print("no completed runs")
            return result, status

        mcf_result, mcf_status = _run_solver(
            "MCF", lambda: solve_mcf(inst), mcf_times, n_runs)
        lp_result, lp_status = _run_solver(
            "LP", lambda: solve_lp(inst, time_limit=LP_TIME_LIMIT), lp_times, n_runs)
        milp_result, milp_status = _run_solver(
            "MILP", lambda: solve_milp(inst, time_limit=MILP_TIME_LIMIT),
            milp_times, n_runs, stop_on_timeout=True)

        # Use median times for reporting (use inf if no completed runs)
        def _median_or_inf(times):
            return statistics.median(times) if times else float("inf")
        milp_med = SolverResult(milp_result.objective, _median_or_inf(milp_times), milp_status)
        lp_med = SolverResult(lp_result.objective, _median_or_inf(lp_times), lp_status)
        mcf_med = SolverResult(mcf_result.objective, _median_or_inf(mcf_times), mcf_status)

        # Verify
        verify_results(inst, milp_med, lp_med, mcf_med)

        rows.append({
            "scale": label,
            "n_pools": n_pools,
            "n_workloads": n_wl,
            "n_assignments": inst.n_assignments,
            "n_assignments_before_filter": inst.n_assignments_before_filter,
            "total_supply_demand": int(inst.capacities.sum()),
            "milp": {
                "status": milp_status,
                "objective": milp_result.objective,
                "median_sec": milp_med.solve_time,
                "all_runs_sec": milp_times,
            },
            "lp": {
                "status": lp_status,
                "objective": lp_result.objective,
                "median_sec": lp_med.solve_time,
                "all_runs_sec": lp_times,
            },
            "mcf": {
                "status": mcf_status,
                "objective": mcf_result.objective,
                "median_sec": mcf_med.solve_time,
                "all_runs_sec": mcf_times,
            },
        })

        # Save incrementally after each scale
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        results_file = out_dir / f"results_seed{seed}_{ts}.json"
        with open(results_file, "w") as f:
            json.dump({"seed": seed, "runs": n_runs, "timestamp": ts, "rows": rows}, f, indent=2)
        print(f"  Results saved to {results_file}")

    # Print final results table
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}\n")
    summary = []
    for row in rows:
        milp_med = SolverResult(row["milp"]["objective"], row["milp"]["median_sec"], row["milp"]["status"])
        lp_med = SolverResult(row["lp"]["objective"], row["lp"]["median_sec"], row["lp"]["status"])
        mcf_med = SolverResult(row["mcf"]["objective"], row["mcf"]["median_sec"], row["mcf"]["status"])
        summary.append({
            "Scale": f"{row['scale']} ({row['n_pools']}x{row['n_workloads']})",
            "Assignments": row["n_assignments"],
            "MILP": format_time(milp_med.solve_time, milp_med.status),
            "LP": format_time(lp_med.solve_time, lp_med.status),
            "MCF": format_time(mcf_med.solve_time, mcf_med.status),
            "LP/MILP": format_speedup(milp_med, lp_med),
            "MCF/MILP": format_speedup(milp_med, mcf_med),
        })
    print(tabulate(summary, headers="keys", tablefmt="simple"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GPU allocation formulation benchmark")
    parser.add_argument(
        "--scales", nargs="+", default=list(SCALES.keys()),
        help=f"Scales to run (choices: {', '.join(SCALES.keys())})",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--runs", type=int, default=N_RUNS, help="Runs per config (default: 3)")
    args = parser.parse_args()

    print(f"GPU Allocation Formulation Benchmark")
    print(f"Seed: {args.seed}, Runs per config: {args.runs}")
    print(f"Scales: {', '.join(args.scales)}")
    print(f"MILP time limit: {MILP_TIME_LIMIT}s")

    run_benchmark(args.scales, args.seed, args.runs)


if __name__ == "__main__":
    main()
