"""Figure: primary crystallite size inside a fixed 8 um secondary particle.

The x axis is the one length that actually sets the rate limit once the
secondary particle is porous and templated.  The upper axis restates it as the
solid-state diffusion time ``r**2 / D``, so the knee in every curve can be read
off directly: it sits where that time crosses the discharge duration.

Run with ``python Sanders_LTO/plot_primary_sweep.py`` after
``run_primary_sweep.py``.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING

import lto_model as lm
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps
import run_primary_sweep as rps
import sanders_data as sd

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

#: C-rates drawn in panel a, from slow to fast (an ordered magnitude, so an
#: ordinal one-hue ramp rather than categorical hues).
PANEL_A_C_RATES = (1.0, 5.0, 10.0, 20.0)

#: Primary diameters drawn as rate curves in panel b.
PANEL_B_DIAMETERS_NM = (50.0, 172.0, 400.0, 1000.0)

SANDERS_PRIMARY_NM = sd.PRIMARY_PARTICLE_Z_AVERAGE_M * 1e9


def load_sweep() -> list[dict[str, str]]:
    """Read ``results/primary_sweep.csv``.

    Returns:
        The sweep rows as dictionaries of strings.
    """
    with (lm.RESULTS_DIR / "primary_sweep.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _series(
    rows: list[dict[str, str]],
    *,
    c_rate: float | None = None,
    diameter_nm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull one curve out of the sweep.

    Args:
        rows: All sweep rows.
        c_rate: Fix the C-rate to sweep diameter.
        diameter_nm: Fix the diameter to sweep C-rate.

    Returns:
        ``(x, specific_capacity)`` where ``x`` is whichever variable is free.
    """
    selected = [
        r
        for r in rows
        if (c_rate is None or np.isclose(float(r["c_rate"]), c_rate))
        and (
            diameter_nm is None
            or np.isclose(float(r["primary_diameter_nm"]), diameter_nm)
        )
    ]
    key = "primary_diameter_nm" if c_rate is not None else "c_rate"
    x = np.array([float(r[key]) for r in selected])
    y = np.array([float(r["specific_capacity_mah_g"]) for r in selected])
    order = np.argsort(x)
    return x[order], y[order]


def panel_capacity_vs_primary(
    axis: Axes, rows: list[dict[str, str]], calibration: lm.Calibration
) -> None:
    """Specific capacity against primary diameter, at four C-rates.

    Args:
        axis: Axes to draw on.
        rows: All sweep rows.
        calibration: Fitted quantities, for the diffusion-time axis.
    """
    for c_rate, color, marker in zip(
        PANEL_A_C_RATES,
        ps.ORDINAL_BLUES,
        ps.MARKERS,
        strict=True,
    ):
        diameters, capacities = _series(rows, c_rate=c_rate)
        axis.plot(
            diameters,
            capacities,
            color=color,
            marker=marker,
            markersize=4.5,
            label=f"{c_rate:g}C",
            zorder=3,
        )
    axis.set_xscale("log")
    axis.set_xlim(20, 1500)
    axis.set_ylim(0, 185)
    ticks = [20, 50, 100, 172, 400, 1000]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(t) for t in ticks])
    axis.minorticks_off()

    axis.axvspan(
        rps.MEMBRANE_BLOCKING_PRIMARY_NM,
        1500,
        color=ps.GRID,
        alpha=0.55,
        zorder=0,
    )
    axis.text(
        1450,
        0.95,
        "blockage risk: no longer far\nsmaller than the 5 µm membrane pores",
        transform=axis.get_xaxis_transform(),
        fontsize=7.3,
        color=ps.TEXT_MUTED,
        ha="right",
        va="top",
    )
    axis.axvline(
        SANDERS_PRIMARY_NM,
        color=ps.TEXT_MUTED,
        linestyle="--",
        linewidth=1.0,
    )
    axis.text(
        SANDERS_PRIMARY_NM * 0.94,
        0.05,
        "Sanders' powder,\n172 nm",
        transform=axis.get_xaxis_transform(),
        fontsize=7.8,
        color=ps.TEXT_MUTED,
        ha="right",
        va="bottom",
    )

    diffusivity = calibration.lto_diffusivity_m2_s

    def to_tau(diameter_nm: ArrayLike) -> np.ndarray:
        """Diameter [nm] -> solid-state diffusion time [s]."""
        return (np.asarray(diameter_nm) * 1e-9 / 2.0) ** 2 / diffusivity

    def from_tau(tau_s: ArrayLike) -> np.ndarray:
        """Solid-state diffusion time [s] -> diameter [nm]."""
        return 2e9 * np.sqrt(np.asarray(tau_s) * diffusivity)

    tau_axis = axis.secondary_xaxis("top", functions=(to_tau, from_tau))
    tau_axis.set_xlabel("Solid-state diffusion time $r^2/D$ / s", fontsize=8.8)
    tau_axis.tick_params(labelsize=8)

    axis.set_xlabel("Primary crystallite diameter / nm")
    axis.set_ylabel("Specific capacity / mAh g$^{-1}$")
    axis.set_title(
        "a   Rate limit set by the primary, not the secondary",
        loc="left",
        pad=34,
    )
    axis.legend(loc="lower left", title="Discharge rate", title_fontsize=8)
    axis.grid(alpha=0.9)
    axis.set_axisbelow(True)


def panel_rate_curves(axis: Axes, rows: list[dict[str, str]]) -> None:
    """Rate curves at four primary sizes, with the measured points over.

    Args:
        axis: Axes to draw on.
        rows: All sweep rows.
    """
    for diameter_nm, color, marker in zip(
        PANEL_B_DIAMETERS_NM,
        ps.ORDINAL_BLUES,
        ps.MARKERS,
        strict=True,
    ):
        rates, capacities = _series(rows, diameter_nm=diameter_nm)
        is_sanders = np.isclose(diameter_nm, SANDERS_PRIMARY_NM)
        axis.plot(
            rates,
            capacities,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=2.6 if is_sanders else 2.0,
            label=(
                f"{diameter_nm:g} nm primary"
                + (" (Sanders' powder)" if is_sanders else "")
            ),
            zorder=3,
        )
    measured = sd.SPECIFIC_CAPACITY_MAH_G["uniform"]
    axis.plot(
        sd.RATE_TEST_C_RATES,
        measured,
        color=ps.SERIES[1],
        marker="o",
        markersize=5,
        linestyle="none",
        label="Measured, membrane emulsified (Table S8)",
        zorder=4,
    )
    axis.set_xscale("log")
    rate_ticks = [0.1, 0.5, 1, 2, 3, 5, 10, 20]
    axis.set_xticks(rate_ticks)
    axis.set_xticklabels([f"{r:g}" for r in rate_ticks])
    axis.minorticks_off()
    axis.set_xlim(0.09, 23)
    axis.set_ylim(0, 185)
    axis.set_xlabel("C-rate")
    axis.set_ylabel("Specific capacity / mAh g$^{-1}$")
    axis.set_title(
        "b   Rate curves at a fixed 8.0 µm secondary particle",
        loc="left",
        pad=34,
    )
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=0.9)
    axis.set_axisbelow(True)


def figure_primary_sweep(
    rows: list[dict[str, str]], calibration: lm.Calibration
) -> Figure:
    """Build the two-panel primary-size figure.

    Args:
        rows: All sweep rows.
        calibration: Fitted model quantities.

    Returns:
        The completed figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7))
    panel_capacity_vs_primary(axes[0], rows, calibration)
    panel_rate_curves(axes[1], rows)
    fig.suptitle(
        "Predicted effect of precursor nanopowder size at Sanders'"
        " 8 µm secondary particle",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005,
        -0.055,
        "Electrode packing, intra-particle porosity and intrinsic"
        " exchange-current density are held fixed; only the diffusion domain"
        " and interfacial area move with the primary size. Sanders et al. use"
        " one 172 nm powder throughout, so everything off that line is"
        " extrapolation.",
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main() -> None:
    """Render and save the primary-size figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ps.use_house_style()
    calibration = lm.Calibration.load()
    rows = load_sweep()
    figure = figure_primary_sweep(rows, calibration)
    path = lm.RESULTS_DIR / "primary_size_sweep.png"
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path)

    diameters, capacities = _series(rows, c_rate=20.0)
    ceiling = capacities.max()
    within_2pct = diameters[capacities >= 0.98 * ceiling]
    logger.info(
        "20C capacity saturates at %.0f mAh/g; within 2 %% of that for"
        " primaries below %.0f nm",
        ceiling,
        within_2pct.max(),
    )
    sanders = float(capacities[np.isclose(diameters, SANDERS_PRIMARY_NM)][0])
    logger.info(
        "Sanders' 172 nm powder reaches %.0f mAh/g at 20C in the model"
        " (%.0f mAh/g measured); a 50 nm powder would reach %.0f mAh/g",
        sanders,
        sd.SPECIFIC_CAPACITY_MAH_G["uniform"][-1],
        float(capacities[np.isclose(diameters, 50.0)][0]),
    )


if __name__ == "__main__":
    main()
