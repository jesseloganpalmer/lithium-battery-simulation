"""Heatmaps of the joint primary x secondary design space.

Four panels: two discharge rates against the two coupling assumptions of
:mod:`run_scale_map`.  Colour is volumetric energy density, a single-hue
sequential ramp because it encodes magnitude, and every cell is also labelled
so the map doubles as a table.

Run with ``python Sanders_LTO/plot_scale_map.py`` after ``run_scale_map.py``.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import lto_model as lm
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps
import run_scale_map as rsm
import sanders_data as sd
from matplotlib.colors import LinearSegmentedColormap, Normalize

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

#: Sequential single-hue ramp, palette blue steps 100 -> 700.
SEQUENTIAL_BLUES = (
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
BLUE_RAMP = LinearSegmentedColormap.from_list("palette_blues", SEQUENTIAL_BLUES)

SANDERS_PRIMARY_NM = sd.PRIMARY_PARTICLE_Z_AVERAGE_M * 1e9
SANDERS_SECONDARY_UM = 8.0

COUPLING_TITLES = {
    "independent": "Scales independent (the model as calibrated)",
    "carbon-limited": "Fixed carbon budget (assumed coupling)",
}


def load_map() -> list[dict[str, str]]:
    """Read ``results/scale_map.csv``.

    Returns:
        The map rows as dictionaries of strings.
    """
    with (lm.RESULTS_DIR / "scale_map.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def grid_for(
    rows: list[dict[str, str]], coupling: str, c_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Reshape one panel's rows into a primary x secondary grid.

    Args:
        rows: All map rows.
        coupling: Coupling mode to select.
        c_rate: C-rate to select.

    Returns:
        ``(energy_density, fits_in_electrode)``, both shaped
        (n_primary, n_secondary) with primary increasing along axis 0.
    """
    energies = np.full(
        (len(rsm.PRIMARY_DIAMETERS_NM), len(rsm.SECONDARY_DIAMETERS_UM)),
        np.nan,
    )
    fits = np.ones_like(energies, dtype=bool)
    for row in rows:
        if row["coupling"] != coupling:
            continue
        if not np.isclose(float(row["c_rate"]), c_rate):
            continue
        i = rsm.PRIMARY_DIAMETERS_NM.index(float(row["primary_diameter_nm"]))
        j = rsm.SECONDARY_DIAMETERS_UM.index(
            float(row["secondary_diameter_um"])
        )
        energies[i, j] = float(row["energy_density_wh_l"])
        fits[i, j] = row["fits_in_electrode"] == "1"
    return energies, fits


def draw_panel(  # noqa: PLR0913
    axis: Axes,
    energies: np.ndarray,
    fits: np.ndarray,
    *,
    norm: Normalize,
    title: str,
    show_ylabel: bool,
    show_xlabel: bool,
) -> None:
    """Draw one heatmap panel.

    Args:
        axis: Axes to draw on.
        energies: Volumetric energy density grid.
        fits: Whether the secondary particle fits inside the coating.
        norm: Shared colour normalisation across all panels.
        title: Panel title.
        show_ylabel: Whether to label the y axis.
        show_xlabel: Whether to label the x axis.
    """
    n_primary, n_secondary = energies.shape
    axis.pcolormesh(
        np.arange(n_secondary + 1),
        np.arange(n_primary + 1),
        energies,
        cmap=BLUE_RAMP,
        norm=norm,
        edgecolors=ps.SURFACE,
        linewidth=1.6,
    )

    for i in range(n_primary):
        for j in range(n_secondary):
            value = energies[i, j]
            dead_cell_wh_l = 1.0
            if value < dead_cell_wh_l:
                axis.text(
                    j + 0.5,
                    i + 0.5,
                    "✕",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=ps.TEXT_MUTED,
                )
                continue
            dark_cell_threshold = 0.62
            axis.text(
                j + 0.5,
                i + 0.5,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "#ffffff"
                    if norm(value) > dark_cell_threshold
                    else ps.TEXT_PRIMARY
                ),
            )
            if not fits[i, j]:
                axis.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        hatch="///",
                        edgecolor=ps.TEXT_MUTED,
                        linewidth=0.0,
                        alpha=0.55,
                    )
                )

    best = np.unravel_index(np.nanargmax(energies), energies.shape)
    # Offset into the corner so the star never hides the cell's own value.
    axis.plot(
        best[1] + 0.83,
        best[0] + 0.79,
        marker="*",
        markersize=12,
        color="#ffffff",
        markeredgecolor=ps.TEXT_PRIMARY,
        markeredgewidth=0.9,
        zorder=5,
    )

    sanders = (
        rsm.PRIMARY_DIAMETERS_NM.index(SANDERS_PRIMARY_NM),
        rsm.SECONDARY_DIAMETERS_UM.index(SANDERS_SECONDARY_UM),
    )
    axis.add_patch(
        plt.Rectangle(
            (sanders[1] + 0.06, sanders[0] + 0.06),
            0.88,
            0.88,
            fill=False,
            edgecolor=ps.SERIES[1],
            linewidth=2.2,
            zorder=4,
        )
    )

    axis.set_xticks(np.arange(n_secondary) + 0.5)
    axis.set_xticklabels([f"{d:g}" for d in rsm.SECONDARY_DIAMETERS_UM])
    axis.set_yticks(np.arange(n_primary) + 0.5)
    axis.set_yticklabels([f"{d:g}" for d in rsm.PRIMARY_DIAMETERS_NM])
    axis.set_title(title, loc="left", fontsize=9.8)
    if show_xlabel:
        axis.set_xlabel("Secondary particle diameter / µm")
    if show_ylabel:
        axis.set_ylabel("Primary crystallite diameter / nm")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)


def figure_scale_map(rows: list[dict[str, str]]) -> Figure:
    """Build the four-panel joint design map.

    Args:
        rows: All map rows.

    Returns:
        The completed figure.
    """
    grids = {
        (coupling, c_rate): grid_for(rows, coupling, c_rate)
        for coupling in rsm.COUPLING_MODES
        for c_rate in rsm.MAP_C_RATES
    }
    finite = np.concatenate([g[g > 0.0].ravel() for g, _ in grids.values()])
    norm = Normalize(vmin=float(finite.min()), vmax=float(finite.max()))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2))
    for row_index, c_rate in enumerate(rsm.MAP_C_RATES):
        for col_index, coupling in enumerate(rsm.COUPLING_MODES):
            energies, fits = grids[(coupling, c_rate)]
            letter = "abcd"[row_index * 2 + col_index]
            draw_panel(
                axes[row_index][col_index],
                energies,
                fits,
                norm=norm,
                title=(f"{letter}   {c_rate:g}C - {COUPLING_TITLES[coupling]}"),
                show_ylabel=col_index == 0,
                show_xlabel=row_index == 1,
            )

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=BLUE_RAMP)
    colorbar = fig.colorbar(
        mappable, ax=axes, fraction=0.035, pad=0.02, aspect=32
    )
    colorbar.set_label("Volumetric energy density / Wh L$^{-1}$", fontsize=9)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(length=0, labelsize=8)

    legend_handles = [
        plt.Line2D(
            [],
            [],
            marker="*",
            markersize=13,
            linestyle="none",
            color="#ffffff",
            markeredgecolor=ps.TEXT_PRIMARY,
            label="Model optimum",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor=ps.SERIES[1],
            linewidth=2.2,
            label="What Sanders et al. made (172 nm x 8 µm)",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            hatch="///",
            edgecolor=ps.TEXT_MUTED,
            linewidth=0.0,
            label="Particle thicker than the coating",
        ),
        plt.Line2D(
            [],
            [],
            marker="$✕$",
            markersize=8,
            linestyle="none",
            color=ps.TEXT_MUTED,
            label="Cell cannot sustain the current",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.005, -0.035),
        ncol=4,
        fontsize=8,
    )

    fig.suptitle(
        "Joint design map: which pair of length scales is best?",
        x=0.004,
        ha="left",
        fontsize=12.5,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.004,
        -0.075,
        "Left column is the model as calibrated, in which the two scales do"
        " not interact. Right column adds a fixed intra-particle carbon"
        " budget, so finer crystallites spread the same carbon thinner - a"
        " plausible but unmeasured coupling, and the only thing that puts a"
        " floor under how fine the primary should be. Read them together:"
        " anything true in one panel and not the other follows from that"
        " assumption, not from the data.",
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        ha="left",
        wrap=True,
    )
    return fig


def main() -> None:
    """Render the joint design map and report its optima."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ps.use_house_style()
    rows = load_map()
    figure = figure_scale_map(rows)
    path = lm.RESULTS_DIR / "scale_map.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    logger.info("wrote %s", path)

    for coupling in rsm.COUPLING_MODES:
        for c_rate in rsm.MAP_C_RATES:
            energies, _ = grid_for(rows, coupling, c_rate)
            best = np.unravel_index(np.nanargmax(energies), energies.shape)
            sanders = energies[
                rsm.PRIMARY_DIAMETERS_NM.index(SANDERS_PRIMARY_NM),
                rsm.SECONDARY_DIAMETERS_UM.index(SANDERS_SECONDARY_UM),
            ]
            logger.info(
                "%-15s %4.0fC: optimum %.0f nm x %.0f um = %.0f Wh/L;"
                " Sanders' 172 nm x 8 um = %.0f Wh/L (%.0f %% of optimum)",
                coupling,
                c_rate,
                rsm.PRIMARY_DIAMETERS_NM[best[0]],
                rsm.SECONDARY_DIAMETERS_UM[best[1]],
                energies[best],
                sanders,
                100.0 * sanders / energies[best],
            )


if __name__ == "__main__":
    main()
