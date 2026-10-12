from __future__ import annotations

import matplotlib.pyplot as plt

# Ground truth from Sanders SI Table S8 (specific capacity, mAh/g) and
# Table S8 caption: "each performing five cycles at the corresponding rate"
rates = [0.1, 0.5, 1, 2, 3, 5, 10, 20]
data = {
    "Unstructured LTO NPs": [
        165.9,
        159.6,
        154.9,
        147.3,
        140.9,
        129.5,
        103.1,
        53.2,
    ],
    "LTO MPs (polydisperse, ~2.3 um)": [
        167.1,
        165.5,
        164.4,
        162.4,
        161.1,
        159.3,
        156.1,
        149.7,
    ],
    "LTO MPs (uniform, ~8 um)": [
        167.9,
        166.4,
        165.0,
        162.6,
        161.2,
        159.2,
        155.4,
        148.2,
    ],
    "Commercial LTO MPs (BTR, spray dried)": [
        166.8,
        166.2,
        165.3,
        163.5,
        161.5,
        156.6,
        144.2,
        118.4,
    ],
}
colors = {
    "Unstructured LTO NPs": "#D85A30",
    "LTO MPs (polydisperse, ~2.3 um)": "#7F77DD",
    "LTO MPs (uniform, ~8 um)": "#1D9E75",
    "Commercial LTO MPs (BTR, spray dried)": "#888780",
}

cycles_per_block = 5
# Table S8 does not separately tabulate the final recovery-to-0.1C block;
# every figure in the paper shows near-full recovery, so the first 0.1C
# value is repeated at the end. This is a REPLICATED assumption, not
# ground truth -- flagged here rather than silently done.
rate_sequence = [*rates, 0.1]

fig, ax = plt.subplots(figsize=(9, 5.5))

for label, caps in data.items():
    full_caps = [*caps, caps[0]]
    cycles, ys = [], []
    cycle = 0
    for cap in full_caps:
        for _ in range(cycles_per_block):
            cycle += 1
            cycles.append(cycle)
            ys.append(cap)
    ax.plot(
        cycles,
        ys,
        marker="o",
        markersize=3.5,
        linewidth=1.6,
        color=colors[label],
        label=label,
    )

n_blocks = len(rate_sequence)
for i in range(1, n_blocks):
    ax.axvline(
        i * cycles_per_block + 0.5, color="gray", linestyle=":", linewidth=0.8
    )
for i, rate in enumerate(rate_sequence):
    x_center = i * cycles_per_block + cycles_per_block / 2 + 0.5
    ax.text(x_center, 178, f"{rate}C", ha="center", fontsize=9)

ax.set_xlim(0, n_blocks * cycles_per_block + 1)
ax.set_ylim(0, 185)
ax.set_xlabel("Cycle number")
ax.set_ylabel("Specific capacity [mAh/g]")
ax.set_title("Sanders et al. ground-truth rate data (SI Table S8)")
ax.legend(loc="lower left", fontsize=8.5)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("lto/results/sanders_stepplot.png", dpi=150)
print("saved")
