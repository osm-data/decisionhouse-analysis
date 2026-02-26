"""Generate publication-quality scaling plot for the vision paper (GPU allocation)."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Load results from most recent benchmark output
results_dir = Path(__file__).parent / "results"
results_files = sorted(results_dir.glob("results_seed42_*.json"))
if results_files:
    results_file = results_files[-1]  # latest timestamp
else:
    results_file = results_dir / "results_seed42.json"  # fallback to old format
print(f"Reading from {results_file}")
with open(results_file) as f:
    data = json.load(f)

# Support both old format (flat keys) and new format (nested)
rows = data["rows"]
if rows and "n_assignments" in rows[0]:
    # New format
    assignments = [row["n_assignments"] for row in rows]
    milp_sec    = [row["milp"]["median_sec"] for row in rows]
    milp_status = [row["milp"]["status"] for row in rows]
    lp_sec      = [row["lp"]["median_sec"] for row in rows]
    mcf_sec     = [row["mcf"]["median_sec"] for row in rows]
else:
    # Old format
    assignments = [row["Assignments"] for row in rows]
    milp_sec    = [row["_milp_sec"] for row in rows]
    milp_status = ["optimal"] * len(rows)
    lp_sec      = [row["_lp_sec"] for row in rows]
    mcf_sec     = [row["_mcf_sec"] for row in rows]

# Separate MILP into plottable vs failed (OOM/timeout) points
milp_ok_x = [a for a, s in zip(assignments, milp_status) if s == "optimal"]
milp_ok_y = [t for t, s in zip(milp_sec, milp_status) if s == "optimal"]
milp_fail_x = [a for a, s in zip(assignments, milp_status) if s != "optimal"]
milp_fail_y = [t for t, s in zip(milp_sec, milp_status) if s != "optimal"]
milp_fail_status = [s for s in milp_status if s != "optimal"]

# Human-readable labels for x-axis
def format_count(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K" if n >= 10_000 else f"{n/1_000:.1f}K"
    return str(n)

labels = [format_count(a) for a in assignments]

# Style — sized for single-column figure in paper
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.linewidth": 0.4,
    "figure.dpi": 300,
})

fig, ax = plt.subplots(figsize=(3.4, 1.2))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(width=0.4, length=3)

# Lines — only plot MILP where it succeeded (failed points omitted, explained in caption)
ax.plot(milp_ok_x, milp_ok_y, "s-", color="#b85450", markersize=4, linewidth=1.3,
        label="Generic MILP", zorder=3)
ax.plot(assignments, lp_sec, "o-", color="#4878a8", markersize=4, linewidth=1.3,
        label="LP relaxation", zorder=3)
ax.plot(assignments, mcf_sec, "^-", color="#5a9e6f", markersize=4, linewidth=1.3,
        label="Min-cost flow", zorder=3)

# Log scales
ax.set_xscale("log")
ax.set_yscale("log")

# Axis labels
ax.set_xlabel("Decision variables (assignments)")
ax.set_ylabel("Solve time")

# X-axis: use assignment counts as ticks
ax.set_xticks(assignments)
ax.set_xticklabels(labels)
ax.xaxis.set_minor_locator(ticker.NullLocator())

# Y-axis: explicit ticks
ax.set_yticks([0.001, 0.1, 10, 180])
ax.set_yticklabels(["1ms", "100ms", "10s", "3min"])
ax.set_ylim(top=1000)

# Speedup annotations at last 4 data points where MILP completed
annot_indices = [i for i in range(len(assignments)) if milp_status[i] == "optimal"]
annot_indices = annot_indices[-4:]  # last 4 successful MILP points
for i in annot_indices:
    speedup = milp_sec[i] / mcf_sec[i]
    # Place label below MCF marker (divide by ~4 in log space)
    y_label = mcf_sec[i] / 4.0
    ax.annotate(
        f"{speedup:.0f}\u00d7",
        xy=(assignments[i], y_label),
        fontsize=6, fontweight="bold", color="#444444",
        ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
        zorder=5,
    )
    # Thin vertical line from MCF marker down to label
    ax.plot(
        [assignments[i], assignments[i]], [mcf_sec[i], y_label * 1.1],
        color="#bbbbbb", linewidth=0.7, linestyle="-", zorder=1,
    )

# Extend x-axis just enough for last label
ax.set_xlim(right=assignments[-1] * 1.8)
# Tighten y-axis to reduce empty space below labels
ax.set_ylim(bottom=8e-5)

# Grid
ax.grid(True, which="major", ls=":", alpha=0.35)

# Legend — inside plot, upper left (data starts low-left, so no overlap)
ax.legend(loc="upper left", framealpha=0.9, ncol=1,
          handlelength=1.5, fontsize=6.5, borderpad=0.3, labelspacing=0.2)

# Tight layout — minimize all margins
fig.tight_layout(pad=0.1)

# Save
out_dir = Path(__file__).parent / "results"
out_dir.mkdir(exist_ok=True)
fig.savefig(out_dir / "formulation_scaling_gpu.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(out_dir / "formulation_scaling_gpu.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
print(f"Saved to {out_dir / 'formulation_scaling_gpu.pdf'}")
print(f"Saved to {out_dir / 'formulation_scaling_gpu.png'}")
