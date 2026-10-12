from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("lto/results/lto_rate_sweep_results.csv")

# Sanders' protocol (Fig 6d / Fig 7b): 5 cycles held at each C-rate, in this
# order, then a final 5-cycle return to 0.1C to check recovery/reversibility.
rate_sequence = [0.1, 0.5, 1, 2, 3, 5, 10, 20, 0.1]
cycles_per_block = 5


def capacity_at_rate(df, diameter_um, c_rate):
    sub = df[df.particle_diameter_um == diameter_um]
    # nearest available C-rate in our simulated grid
    idx = (sub["C_rate"] - c_rate).abs().idxmin()
    return sub.loc[idx, "specific_capacity_mAh_g"]


def build_step_series(df, diameter_um):
    cycles, caps = [], []
    cycle = 0
    for rate in rate_sequence:
        cap = capacity_at_rate(df, diameter_um, rate)
        for _ in range(cycles_per_block):
            cycle += 1
            cycles.append(cycle)
            caps.append(cap)
    return np.array(cycles), np.array(caps)


fig, ax = plt.subplots(figsize=(9, 5.5))

# Plot every simulated secondary particle size as its own step curve, labelled
# purely by size (no "structured"/"unstructured" framing -- that binary split
# was Sanders' experimental categories, not what this model varies; the model
# varies a continuous size parameter and should be presented that way).
sizes_um = sorted(df["particle_diameter_um"].unique())
cmap = plt.cm.viridis_r(np.linspace(0, 1, len(sizes_um)))

for d_um, color in zip(sizes_um, cmap, strict=True):
    cycles, caps = build_step_series(df, d_um)
    ax.plot(
        cycles,
        caps,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        color=color,
        label=f"d = {d_um} \u03bcm",
    )

# Vertical dividers + C-rate labels between rate blocks, matching Sanders' style
n_blocks = len(rate_sequence)
for i in range(1, n_blocks):
    ax.axvline(
        i * cycles_per_block + 0.5, color="gray", linestyle=":", linewidth=0.8
    )

for i, rate in enumerate(rate_sequence):
    x_center = i * cycles_per_block + cycles_per_block / 2 + 0.5
    ax.text(x_center, 205, f"{rate}C", ha="center", fontsize=9)

ax.set_xlim(0, n_blocks * cycles_per_block + 1)
ax.set_ylim(0, 215)
ax.set_xlabel("Cycle number")
ax.set_ylabel("Specific capacity [mAh/g]")
ax.set_title(
    "PyBaMM (ensemble model): rate test across secondary particle sizes"
)
ax.legend(loc="lower left", fontsize=8, ncol=3, framealpha=0.9)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("lto/results/rate_performance_stepplot.png", dpi=150)
print("saved")
