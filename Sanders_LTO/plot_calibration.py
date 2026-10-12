"""Model-versus-experiment figures for the calibrated LTO model.

Two things are shown:

* the simulated rate curves laid over the measured ones, in the same
  step-plot form as the paper's Figure 6d;
* the fitted effective lithium diffusion domain radius of every electrode,
  which is the model's one structure-sensitive parameter.  The second panel
  is the real result: for every droplet-templated sample the fit returns the
  independently measured 86 nm primary crystallite radius, from rate data
  alone and from a search range spanning 40 nm to 3 µm, whereas the
  unstructured nanopowder comes back around three times larger.

Run with ``python Sanders_LTO/plot_calibration.py`` after ``calibrate.py``.
"""

from __future__ import annotations

import csv
import logging

import lto_model as lm
import matplotlib.pyplot as plt
import numpy as np
import plot_style as ps
import sanders_data as sd

logger = logging.getLogger(__name__)

OVERLAY_KEYS = ("uniform", "polydisperse", "nano")
OVERLAY_LABELS = {
    "uniform": "Membrane emulsified (8.0 µm)",
    "polydisperse": "Bulk emulsified (2.3 µm)",
    "nano": "Unstructured nanoparticles",
}


def simulate_overlay_curves(
    calibration: lm.Calibration,
) -> dict[str, np.ndarray]:
    """Simulate the rate curves of the three pure-LTO electrodes.

    Args:
        calibration: Fitted model quantities.

    Returns:
        A mapping from electrode key to simulated specific capacity per rate.
    """
    curves = {}
    for key in OVERLAY_KEYS:
        structure = lm.structure_from_electrode(key, calibration)
        curves[key] = lm.rate_curve(structure, calibration)[
            "specific_capacity_mah_g"
        ]
    return curves


def figure_model_vs_experiment(
    curves: dict[str, np.ndarray],
    fitted_rows: list[dict[str, str]],
) -> plt.Figure:
    """Build the two-panel model-versus-experiment figure.

    Args:
        curves: Simulated capacities from :func:`simulate_overlay_curves`.
        fitted_rows: Rows of ``results/fitted_diffusion_radii.csv``.

    Returns:
        The completed figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # --- Panel a: measured vs simulated rate curves ------------------------
    axis = axes[0]
    for key, color in zip(OVERLAY_KEYS, ps.SERIES, strict=True):
        ps.plot_step_curve(
            axis,
            ps.with_recovery(sd.SPECIFIC_CAPACITY_MAH_G[key]),
            deviations=ps.with_recovery(sd.SPECIFIC_CAPACITY_SD[key]),
            label=f"{OVERLAY_LABELS[key]} - measured",
            color=color,
            marker="o",
        )
        ps.plot_step_curve(
            axis,
            ps.with_recovery(curves[key]),
            label=f"{OVERLAY_LABELS[key]} - PyBaMM",
            color=color,
            marker="",
            linestyle="--",
        )
    ps.annotate_rate_blocks(axis, ps.RATES_WITH_RECOVERY)
    axis.set_ylabel("Specific capacity / mAh g$^{-1}$")
    axis.set_title("a   Calibrated model against Table S8", loc="left")
    axis.set_ylim(0, 200)
    axis.legend(loc="lower left", ncol=1, fontsize=7.6)

    # --- Panel b: the fitted diffusion domain radius ------------------------
    axis = axes[1]
    rows = sorted(
        fitted_rows, key=lambda r: float(r["fitted_diffusion_radius_nm"])
    )
    positions = np.arange(len(rows))
    radii = np.array([float(r["fitted_diffusion_radius_nm"]) for r in rows])
    is_templated = np.array(
        [float(r["secondary_diameter_um"]) > 0.0 for r in rows]
    )
    colors = [
        ps.SERIES[0] if templated else ps.SERIES[2]
        for templated in is_templated
    ]
    axis.barh(
        positions,
        radii,
        color=colors,
        height=0.62,
        edgecolor=ps.SURFACE,
        linewidth=1.5,
    )
    primary_nm = sd.PRIMARY_PARTICLE_Z_AVERAGE_M / 2.0 * 1e9
    axis.axvline(
        primary_nm, color=ps.TEXT_SECONDARY, linewidth=1.4, linestyle=":"
    )
    axis.annotate(
        f"measured primary radius, {primary_nm:.0f} nm",
        xy=(primary_nm, -0.75),
        xytext=(primary_nm + 55, -1.05),
        fontsize=7.8,
        color=ps.TEXT_SECONDARY,
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": ps.TEXT_SECONDARY,
            "linewidth": 0.8,
        },
    )
    axis.set_ylim(-1.45, len(rows) - 0.4)
    for position, radius in zip(positions, radii, strict=True):
        axis.text(
            radius + 4,
            position,
            f"{radius:.0f} nm",
            va="center",
            fontsize=8,
            color=ps.TEXT_SECONDARY,
        )
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [_short_label(r["electrode"]) for r in rows], fontsize=8.2
    )
    axis.set_xlabel("Fitted effective Li diffusion domain radius / nm")
    axis.set_xlim(0, max(radii) * 1.32)
    axis.set_title("b   Fitted diffusion domain radius", loc="left")
    axis.grid(axis="x", alpha=0.9)
    axis.set_axisbelow(True)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ps.SERIES[0]),
        plt.Rectangle((0, 0), 1, 1, color=ps.SERIES[2]),
    ]
    axis.legend(
        handles,
        ["Droplet-templated secondary particles", "Unstructured nanopowder"],
        loc="lower right",
        fontsize=8,
    )

    fig.suptitle(
        "The templated microparticles diffuse over their primary"
        " crystallites; the loose nanopowder does not",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005,
        -0.03,
        "PyBaMM DFN half cell. One global diffusivity plus one radius per"
        " electrode are fitted; loadings, thicknesses, porosities and EIS"
        " resistances are taken from Sanders' Tables S6 and S10.",
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def _short_label(key: str) -> str:
    """Return a compact axis label for an electrode key.

    Args:
        key: Key into :data:`sanders_data.ELECTRODES`.

    Returns:
        A short human-readable label.

    Examples:
        >>> _short_label("uniform")
        'Membrane LTO (8.0 µm)'
    """
    return {
        "nano": "Unstructured nLTO",
        "nano_oxcnt": "Unstructured nLTO/oxCNT",
        "nano_cnt": "Unstructured nLTO/CNT",
        "polydisperse": "Bulk LTO (2.3 µm)",
        "uniform": "Membrane LTO (8.0 µm)",
        "oxcnt_mp": "Membrane LTO/oxCNT (9.1 µm)",
        "cnt_pvp_mp": "Membrane LTO/CNT/PVP (9.3 µm)",
        "commercial": "Commercial spray dried",
    }[key]


def main() -> None:
    """Render and save the model-versus-experiment figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ps.use_house_style()
    calibration = lm.Calibration.load()
    fitted_file = lm.RESULTS_DIR / "fitted_diffusion_radii.csv"
    with fitted_file.open(newline="") as handle:
        fitted_rows = list(csv.DictReader(handle))

    curves = simulate_overlay_curves(calibration)
    figure = figure_model_vs_experiment(curves, fitted_rows)
    path = lm.RESULTS_DIR / "model_vs_experiment.png"
    figure.savefig(path)
    plt.close(figure)
    logger.info("wrote %s", path)

    out = lm.RESULTS_DIR / "model_vs_experiment.csv"
    with out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["electrode", "c_rate", "measured_mah_g", "simulated_mah_g"]
        )
        for key in OVERLAY_KEYS:
            for rate, measured, simulated in zip(
                sd.RATE_TEST_C_RATES,
                sd.SPECIFIC_CAPACITY_MAH_G[key],
                curves[key],
                strict=True,
            ):
                writer.writerow([key, rate, measured, round(simulated, 2)])
    logger.info("wrote %s", out)


if __name__ == "__main__":
    main()
