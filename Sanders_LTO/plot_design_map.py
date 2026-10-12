"""Design maps from the forward sweep: how big should the microparticle be?

Panel a is the result the paper cannot get experimentally, because it would
mean making a dozen more particle sizes: volumetric energy density at 20C as a
function of secondary particle diameter.  It has an optimum, and the optimum
moves with how conductive the carbon network *inside* the particle is - which
is precisely the knob Sanders et al. turn by co-assembling CNTs with the LTO.

Panels b and c show the rate curves behind that map, for the thin coin-cell
electrodes of Figure 6 and for a thick electrode at the loading of Figure S14.

Run with ``python Sanders_LTO/plot_design_map.py`` after ``run_design_sweep``.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import lto_model as lm
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps
import sanders_data as sd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

BASE_CONDUCTIVITY_S_M = 1e-3
THIN_LOADING_MG_CM2 = 1.22
THICK_LOADING_MG_CM2 = 15.5
HIGHLIGHT_DIAMETERS_UM = (2.0, 8.0, 20.0)
SANDERS_DIAMETER_UM = 8.0

CONDUCTIVITY_LABELS = {
    1e-4: "10$^{-4}$ S m$^{-1}$ (poorly connected)",
    1e-3: "10$^{-3}$ S m$^{-1}$ (pyrolysed carbon only)",
    1e-2: "10$^{-2}$ S m$^{-1}$ (CNT co-assembled)",
}


def load_sweep() -> list[dict[str, str]]:
    """Read ``results/design_sweep.csv``.

    Returns:
        The sweep rows as dictionaries of strings.
    """
    with (lm.RESULTS_DIR / "design_sweep.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _select(  # noqa: PLR0913
    rows: list[dict[str, str]],
    *,
    structure: str,
    loading: float,
    conductivity: float | None = None,
    diameter: float | None = None,
    c_rate: float | None = None,
) -> list[dict[str, str]]:
    """Filter sweep rows on the sweep's independent variables.

    Args:
        rows: All sweep rows.
        structure: ``"templated"`` or ``"unstructured"``.
        loading: Active-material areal loading in mg/cm2.
        conductivity: Intra-particle conductivity, if constrained.
        diameter: Secondary particle diameter, if constrained.
        c_rate: C-rate, if constrained.

    Returns:
        The matching rows, in file order.
    """
    out = []
    for row in rows:
        if row["structure"] != structure:
            continue
        if not np.isclose(float(row["loading_mg_cm2"]), loading):
            continue
        if conductivity is not None and not np.isclose(
            float(row["intra_particle_conductivity_s_m"]), conductivity
        ):
            continue
        if diameter is not None and not np.isclose(
            float(row["diameter_um"]), diameter
        ):
            continue
        if c_rate is not None and not np.isclose(float(row["c_rate"]), c_rate):
            continue
        out.append(row)
    return out


def _shade_oversized(axis: Axes, rows: list[dict[str, str]]) -> None:
    """Grey out diameters larger than the coating that would contain them.

    Args:
        axis: Axes to shade.
        rows: The rows plotted on this axis, used to find the cut-off.
    """
    oversized = [
        float(r["diameter_um"]) for r in rows if r["fits_in_electrode"] == "0"
    ]
    if not oversized:
        return
    start = min(oversized)
    axis.axvspan(start, axis.get_xlim()[1], color=ps.GRID, alpha=0.55, zorder=0)
    axis.text(
        start * 1.06,
        0.55,
        "particle thicker\nthan the coating",
        transform=axis.get_xaxis_transform(),
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        va="bottom",
    )


def panel_design_map(axis: Axes, rows: list[dict[str, str]]) -> None:
    """Volumetric energy density at 20C against secondary particle diameter.

    Args:
        axis: Axes to draw on.
        rows: All sweep rows.
    """
    plotted: list[dict[str, str]] = []
    conductivities = sorted(CONDUCTIVITY_LABELS)
    for conductivity, color, marker in zip(
        conductivities,
        ps.SERIES,
        ps.MARKERS[: len(conductivities)],
        strict=True,
    ):
        selected = _select(
            rows,
            structure="templated",
            loading=THIN_LOADING_MG_CM2,
            conductivity=conductivity,
            c_rate=20.0,
        )
        plotted.extend(selected)
        diameters = np.array([float(r["diameter_um"]) for r in selected])
        energies = np.array([float(r["energy_density_wh_l"]) for r in selected])
        axis.plot(
            diameters,
            energies,
            color=color,
            marker=marker,
            markersize=4.5,
            label=CONDUCTIVITY_LABELS[conductivity],
            zorder=3,
        )
        peak = int(np.argmax(energies))
        axis.plot(
            diameters[peak],
            energies[peak],
            marker="*",
            markersize=13,
            color=color,
            markeredgecolor=ps.SURFACE,
            markeredgewidth=1.2,
            zorder=4,
        )

    baseline = _select(
        rows,
        structure="unstructured",
        loading=THIN_LOADING_MG_CM2,
        c_rate=20.0,
    )[0]
    axis.axhline(
        float(baseline["energy_density_wh_l"]),
        color=ps.TEXT_SECONDARY,
        linestyle=":",
        linewidth=1.4,
    )
    axis.text(
        1.05,
        float(baseline["energy_density_wh_l"]) + 2,
        "unstructured nanopowder",
        fontsize=7.8,
        color=ps.TEXT_SECONDARY,
    )
    axis.set_xscale("log")
    axis.set_xlim(1.0, 100.0)
    axis.set_ylim(0, 100)
    size_ticks = [1, 2, 5, 10, 20, 50, 100]
    axis.set_xticks(size_ticks)
    axis.set_xticklabels([str(t) for t in size_ticks])
    axis.minorticks_off()
    axis.axvline(
        SANDERS_DIAMETER_UM,
        color=ps.TEXT_MUTED,
        linewidth=1.0,
        linestyle="--",
    )
    axis.text(
        SANDERS_DIAMETER_UM * 1.08,
        0.93,
        "Sanders et al.\nmade 8.0 µm",
        transform=axis.get_xaxis_transform(),
        fontsize=7.8,
        color=ps.TEXT_MUTED,
        va="top",
    )
    _shade_oversized(axis, plotted)
    axis.set_xlabel("Secondary particle diameter / µm")
    axis.set_ylabel("Energy density at 20C / Wh L$^{-1}$")
    axis.set_title(
        "a   Optimal size, set by intra-particle conductivity", loc="left"
    )
    axis.legend(loc="lower center", fontsize=8)
    axis.grid(alpha=0.9)
    axis.set_axisbelow(True)


def panel_rate_curves(
    axis: Axes,
    rows: list[dict[str, str]],
    *,
    loading: float,
    max_c_rate: float,
    title: str,
) -> None:
    """Energy density against C-rate for three representative diameters.

    Args:
        axis: Axes to draw on.
        rows: All sweep rows.
        loading: Active-material areal loading in mg/cm2.
        max_c_rate: Largest C-rate to show.
        title: Panel title.
    """
    ramp = ps.ORDINAL_BLUES[1:]
    for diameter, color in zip(HIGHLIGHT_DIAMETERS_UM, ramp, strict=True):
        selected = _select(
            rows,
            structure="templated",
            loading=loading,
            conductivity=BASE_CONDUCTIVITY_S_M,
            diameter=diameter,
        )
        rates = np.array([float(r["c_rate"]) for r in selected])
        energies = np.array([float(r["energy_density_wh_l"]) for r in selected])
        keep = rates <= max_c_rate
        axis.plot(
            rates[keep],
            energies[keep],
            color=color,
            marker="o",
            markersize=4.5,
            label=f"Templated, {diameter:g} µm",
            zorder=3,
        )

    baseline = _select(rows, structure="unstructured", loading=loading)
    rates = np.array([float(r["c_rate"]) for r in baseline])
    energies = np.array([float(r["energy_density_wh_l"]) for r in baseline])
    keep = rates <= max_c_rate
    axis.plot(
        rates[keep],
        energies[keep],
        color=ps.SERIES[2],
        marker="^",
        markersize=4.5,
        linestyle="--",
        label="Unstructured nanopowder",
        zorder=3,
    )
    axis.set_xscale("log")
    axis.set_xlabel("C-rate")
    axis.set_ylabel("Energy density / Wh L$^{-1}$")
    axis.set_title(title, loc="left")
    axis.set_ylim(0, 120)
    axis.set_xticks([0.1, 0.5, 1, 2, 3, 5, 10, 20])
    axis.set_xticklabels([f"{r:g}" for r in [0.1, 0.5, 1, 2, 3, 5, 10, 20]])
    axis.set_xlim(0.09, max_c_rate * 1.15)
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.9)
    axis.set_axisbelow(True)


def figure_design_map(rows: list[dict[str, str]]) -> Figure:
    """Assemble the three-panel design map.

    Args:
        rows: All sweep rows.

    Returns:
        The completed figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.5))
    panel_design_map(axes[0], rows)
    panel_rate_curves(
        axes[1],
        rows,
        loading=THIN_LOADING_MG_CM2,
        max_c_rate=20.0,
        title=(
            f"b   Thin electrode, {THIN_LOADING_MG_CM2:g} mg cm$^{{-2}}$"
            " (as Figure 6)"
        ),
    )
    panel_rate_curves(
        axes[2],
        rows,
        loading=THICK_LOADING_MG_CM2,
        max_c_rate=3.0,
        title=(
            f"c   Thick electrode, {THICK_LOADING_MG_CM2:g} mg cm$^{{-2}}$"
            " (as Figure S14)"
        ),
    )
    fig.suptitle(
        "Predicted design map for droplet-templated LTO microparticles",
        x=0.004,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.004,
        -0.03,
        "PyBaMM DFN half cell calibrated on Sanders et al.'s three measured"
        " LTO electrodes, then extrapolated. Energy densities are per"
        " electrode volume (loading / cast density), which runs about 1.6x"
        " below the paper's tabulated Wh/L; compare trends, not absolutes.",
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    """Render and save the design-map figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ps.use_house_style()
    rows = load_sweep()
    figure = figure_design_map(rows)
    path = lm.RESULTS_DIR / "design_map.png"
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path)

    peaks = []
    for conductivity in sorted(CONDUCTIVITY_LABELS):
        selected = _select(
            rows,
            structure="templated",
            loading=THIN_LOADING_MG_CM2,
            conductivity=conductivity,
            c_rate=20.0,
        )
        best = max(selected, key=lambda r: float(r["energy_density_wh_l"]))
        peaks.append(
            (
                conductivity,
                float(best["diameter_um"]),
                float(best["energy_density_wh_l"]),
            )
        )
        logger.info(
            "sigma_intra = %.0e S/m -> optimum d = %.0f um, %.0f Wh/L at 20C",
            *peaks[-1],
        )
    baseline = _select(
        rows,
        structure="unstructured",
        loading=THIN_LOADING_MG_CM2,
        c_rate=20.0,
    )[0]
    logger.info(
        "unstructured baseline at 20C: %.0f Wh/L (%.0f mAh/g)",
        float(baseline["energy_density_wh_l"]),
        float(baseline["specific_capacity_mah_g"]),
    )
    logger.info(
        "Sanders measured %.0f Wh/L at 20C for their 8.0 um particles"
        " and %.0f Wh/L unstructured (Table S9).",
        sd.ENERGY_DENSITY_WH_L["uniform"][-1],
        sd.ENERGY_DENSITY_WH_L["nano"][-1],
    )


if __name__ == "__main__":
    main()
