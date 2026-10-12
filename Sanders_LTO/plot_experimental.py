"""Replot Sanders et al.'s rate-test step figures from the tabulated data.

Reproduces, from the Supporting Information tables rather than from pixels:

* Figure 6d/6e - specific capacity and volumetric energy density of
  unstructured, bulk-emulsified and membrane-emulsified LTO;
* Figure 7b/7c - the same for the LTO/CNT composites, structured vs
  unstructured;
* Figure S12a/b - the commercial spray-dried benchmark;
* Figure S14a/b - the thick, high-loading electrode.

Run with ``python Sanders_LTO/plot_experimental.py``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import plot_style as ps
import sanders_data as sd
from lto_model import RESULTS_DIR

if TYPE_CHECKING:
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

SOURCE_NOTE = (
    "Data: Sanders et al., ACS Nano 2026, 20, 7555 (CC-BY 4.0), "
    "Tables S8/S9. Final 0.1C block reconstructed from the first."
)


def _draw_panel(  # noqa: PLR0913
    axis: Axes,
    keys: tuple[str, ...],
    table: dict[str, tuple[float, ...]],
    deviations: dict[str, tuple[float, ...]],
    labels: dict[str, str],
    colors: tuple[str, ...],
    *,
    ylabel: str,
    title: str,
    styles: tuple[tuple[str, str, bool], ...] | None = None,
) -> None:
    """Draw one rate-test step panel.

    Args:
        axis: Axes to draw on.
        keys: Electrode keys to plot.
        table: Mapping of electrode key to per-rate values.
        deviations: Mapping of electrode key to per-rate standard deviations.
        labels: Legend label per electrode key.
        colors: Colour per series, same length as ``keys``.
        ylabel: Y axis label.
        title: Panel title.
        styles: Optional ``(marker, linestyle, filled)`` per series, used when
            two series share a hue.
    """
    if styles is None:
        styles = tuple(
            (ps.MARKERS[i % len(ps.MARKERS)], "-", True)
            for i in range(len(keys))
        )
    for key, color, (marker, linestyle, filled) in zip(
        keys, colors, styles, strict=True
    ):
        ps.plot_step_curve(
            axis,
            ps.with_recovery(table[key]),
            deviations=ps.with_recovery(deviations[key]),
            label=labels[key],
            color=color,
            marker=marker,
            linestyle=linestyle,
            filled=filled,
        )
    ps.annotate_rate_blocks(axis, ps.RATES_WITH_RECOVERY)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    axis.legend(loc="lower left", ncol=1)


def figure_lto_rate_tests() -> plt.Figure:
    """Replot Figure 6d and 6e: structuring effect on the pure LTO electrodes.

    Returns:
        The completed figure.
    """
    keys = ("uniform", "polydisperse", "nano")
    labels = {
        "uniform": "Membrane emulsified, uniform (8.0 µm, CV 17 %)",
        "polydisperse": "Bulk emulsified, polydisperse (2.3 µm, CV 67 %)",
        "nano": "Unstructured nanoparticles (no secondary particle)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    _draw_panel(
        axes[0],
        keys,
        sd.SPECIFIC_CAPACITY_MAH_G,
        sd.SPECIFIC_CAPACITY_SD,
        labels,
        ps.SERIES,
        ylabel="Specific capacity / mAh g$^{-1}$",
        title="a   Gravimetric capacity",
    )
    axes[0].set_ylim(0, 200)
    _draw_panel(
        axes[1],
        keys,
        sd.ENERGY_DENSITY_WH_L,
        sd.ENERGY_DENSITY_SD,
        labels,
        ps.SERIES,
        ylabel="Energy density / Wh L$^{-1}$",
        title="b   Volumetric energy density",
    )
    axes[1].set_ylim(0, 210)
    fig.suptitle(
        "Secondary particle structuring controls rate performance"
        " (Sanders et al., Figure 6d/6e)",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005, -0.03, SOURCE_NOTE, fontsize=7.5, color=ps.TEXT_MUTED, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_composite_rate_tests() -> plt.Figure:
    """Replot Figure 7b and 7c: CNT composites, structured vs unstructured.

    Two hues carry the CNT chemistry and the line style carries whether the
    CNTs are locked inside secondary particles or loose in the electrode, so
    four series are distinguishable without a fourth hue.

    Returns:
        The completed figure.
    """
    keys = ("oxcnt_mp", "nano_oxcnt", "cnt_pvp_mp", "nano_cnt")
    labels = {
        "oxcnt_mp": "LTO/oxCNT microparticles (structured)",
        "nano_oxcnt": "LTO + oxCNT powder mix (unstructured)",
        "cnt_pvp_mp": "LTO/CNT/PVP microparticles (structured)",
        "nano_cnt": "LTO + CNT powder mix (unstructured)",
    }
    colors = (ps.SERIES[0], ps.SERIES[0], ps.SERIES[1], ps.SERIES[1])
    styles = (
        ("o", "-", True),
        ("o", "--", False),
        ("s", "-", True),
        ("s", "--", False),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    _draw_panel(
        axes[0],
        keys,
        sd.SPECIFIC_CAPACITY_MAH_G,
        sd.SPECIFIC_CAPACITY_SD,
        labels,
        colors,
        ylabel="Specific capacity / mAh g$^{-1}$",
        title="a   Gravimetric capacity",
        styles=styles,
    )
    axes[0].set_ylim(0, 200)
    _draw_panel(
        axes[1],
        keys,
        sd.ENERGY_DENSITY_WH_L,
        sd.ENERGY_DENSITY_SD,
        labels,
        colors,
        ylabel="Energy density / Wh L$^{-1}$",
        title="b   Volumetric energy density",
        styles=styles,
    )
    axes[1].set_ylim(0, 210)
    fig.suptitle(
        "Locking CNTs inside the secondary particle, not beside it"
        " (Sanders et al., Figure 7b/7c)",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005, -0.03, SOURCE_NOTE, fontsize=7.5, color=ps.TEXT_MUTED, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_commercial_benchmark() -> plt.Figure:
    """Replot Figure S12a/b against the membrane-emulsified material.

    Returns:
        The completed figure.
    """
    keys = ("uniform", "commercial")
    labels = {
        "uniform": "Membrane emulsified (8.0 µm, CV 17 %)",
        "commercial": "Commercial spray dried (BTR, D$_{50}$ 4-10 µm)",
    }
    colors = (ps.SERIES[0], ps.BENCHMARK)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    _draw_panel(
        axes[0],
        keys,
        sd.SPECIFIC_CAPACITY_MAH_G,
        sd.SPECIFIC_CAPACITY_SD,
        labels,
        colors,
        ylabel="Specific capacity / mAh g$^{-1}$",
        title="a   Gravimetric capacity",
    )
    axes[0].set_ylim(100, 185)
    _draw_panel(
        axes[1],
        keys,
        sd.ENERGY_DENSITY_WH_L,
        sd.ENERGY_DENSITY_SD,
        labels,
        colors,
        ylabel="Energy density / Wh L$^{-1}$",
        title="b   Volumetric energy density",
    )
    axes[1].set_ylim(100, 200)
    fig.suptitle(
        "Uniform droplet-templated LTO overtakes the spray-dried benchmark"
        " above 5C (Sanders et al., Figure S12)",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005, -0.03, SOURCE_NOTE, fontsize=7.5, color=ps.TEXT_MUTED, ha="left"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def figure_high_loading() -> plt.Figure:
    """Replot Figure S14a/b: the 19.4 mg/cm2 microparticle electrode.

    Returns:
        The completed figure.
    """
    rates = sd.HIGH_LOADING_C_RATES
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))
    panels = (
        (
            axes[0],
            "specific_capacity_mah_g",
            "specific_capacity_sd",
            "Specific capacity / mAh g$^{-1}$",
            "a   Gravimetric capacity",
            (0, 175),
        ),
        (
            axes[1],
            "energy_density_wh_l",
            "energy_density_sd",
            "Energy density / Wh L$^{-1}$",
            "b   Volumetric energy density",
            (0, 275),
        ),
    )
    for axis, value_key, sd_key, ylabel, title, ylim in panels:
        ps.plot_step_curve(
            axis,
            ps.with_recovery(sd.HIGH_LOADING[value_key]),
            deviations=ps.with_recovery(sd.HIGH_LOADING[sd_key]),
            label=(
                "Membrane-emulsified LTO microparticles, "
                f"{sd.HIGH_LOADING['areal_loading_mg_cm2']:.1f} mg cm$^{{-2}}$"
            ),
            color=ps.SERIES[0],
        )
        ps.annotate_rate_blocks(axis, (*rates, 0.1))
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.set_ylim(*ylim)
        axis.legend(loc="lower left")
    fig.suptitle(
        "Thick electrode from uniform microparticles, 245 µm and"
        " crack-free (Sanders et al., Figure S14)",
        x=0.005,
        ha="left",
        fontsize=12,
        color=ps.TEXT_PRIMARY,
    )
    fig.text(
        0.005,
        -0.03,
        "Data: Sanders et al., Table S15. Final 0.1C block reconstructed"
        " from the first.",
        fontsize=7.5,
        color=ps.TEXT_MUTED,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> None:
    """Render and save every experimental replot."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ps.use_house_style()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "exp_fig6_lto_rate_tests.png": figure_lto_rate_tests,
        "exp_fig7_composite_rate_tests.png": figure_composite_rate_tests,
        "exp_figS12_commercial_benchmark.png": figure_commercial_benchmark,
        "exp_figS14_high_loading.png": figure_high_loading,
    }
    for filename, builder in figures.items():
        figure = builder()
        path = RESULTS_DIR / filename
        figure.savefig(path)
        plt.close(figure)
        logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
