"""Shared figure styling and step-plot construction.

Colours are taken from a validated categorical palette.  Its documented
guarantee is that the first three slots are separable for viewers with colour
vision deficiency across *all* pairs, so no panel here uses more than three
hues for identity; where four series are needed they are encoded as two hues
crossed with a line style, and every series is also named in the legend so
identity is never carried by colour alone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import sanders_data as sd

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes

#: Categorical slots 1-3 of the reference palette (light mode).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
#: Slot 8, reserved here for the single "benchmark" series where one is shown
#: alongside at most two others.
BENCHMARK = "#e34948"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#7a7873"
GRID = "#e4e3df"

#: Ordinal blue ramp (steps 250 / 400 / 550 / 700), for ordered magnitude such
#: as increasing secondary particle diameter.  The lightest step clears the
#: 2:1 contrast floor on a light surface.
ORDINAL_BLUES = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")

MARKERS = ("o", "s", "^", "D")


def use_house_style() -> None:
    """Apply the shared Matplotlib style: recessive axes, thin marks."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titleweight": "medium",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "legend.labelcolor": TEXT_SECONDARY,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.0,
            "font.size": 9.5,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def step_series(
    values: Sequence[float],
    *,
    cycles_per_block: int = sd.CYCLES_PER_RATE_BLOCK,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand one value per C-rate block into a cycle-by-cycle series.

    Sanders' rate-test figures plot cycle number on the x axis with five
    cycles held at each rate; the Supporting Information tabulates only the
    block averages, so each average is repeated across its block.

    Args:
        values: One value per rate block, in the order the blocks were run.
        cycles_per_block: Cycles held at each rate.

    Returns:
        A tuple of ``(cycle_numbers, repeated_values)``.

    Examples:
        >>> cycles, ys = step_series([1.0, 2.0], cycles_per_block=2)
        >>> cycles.tolist(), ys.tolist()
        ([1, 2, 3, 4], [1.0, 1.0, 2.0, 2.0])
    """
    repeated = np.repeat(np.asarray(values, dtype=float), cycles_per_block)
    cycles = np.arange(1, len(repeated) + 1)
    return cycles, repeated


def annotate_rate_blocks(
    axis: Axes,
    rates: Sequence[float],
    *,
    cycles_per_block: int = sd.CYCLES_PER_RATE_BLOCK,
    label_y: float = 0.955,
    recovery_block: bool = True,
) -> None:
    """Draw the C-rate block dividers and their labels.

    Args:
        axis: Axes to annotate.
        rates: The C-rates of each block, in order.
        cycles_per_block: Cycles held at each rate.
        label_y: Vertical position of the block labels, in axes coordinates.
        recovery_block: Whether the final block is the return-to-0.1C
            recovery step, which is labelled separately.
    """
    n_blocks = len(rates)
    for index in range(1, n_blocks):
        axis.axvline(
            index * cycles_per_block + 0.5,
            color=GRID,
            linewidth=0.7,
            zorder=0,
        )
    for index, rate in enumerate(rates):
        centre = index * cycles_per_block + cycles_per_block / 2 + 0.5
        is_recovery = recovery_block and index == n_blocks - 1
        axis.text(
            centre,
            label_y,
            f"{rate:g}C",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color=TEXT_MUTED if is_recovery else TEXT_SECONDARY,
        )
    axis.set_xlim(0.5, n_blocks * cycles_per_block + 0.5)
    axis.set_xlabel("Cycle number")
    axis.grid(axis="y", alpha=0.9, zorder=0)
    axis.set_axisbelow(True)


def plot_step_curve(  # noqa: PLR0913
    axis: Axes,
    values: Sequence[float],
    *,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
    filled: bool = True,
    deviations: Sequence[float] | None = None,
    cycles_per_block: int = sd.CYCLES_PER_RATE_BLOCK,
) -> None:
    """Draw one material's rate-test step curve, with an optional error band.

    Args:
        axis: Axes to draw on.
        values: One value per rate block.
        label: Legend label.
        color: Series colour.
        marker: Marker shape.
        linestyle: Line style, used as the secondary encoding when two series
            share a hue.
        filled: Whether markers are filled (structured) or open
            (unstructured), the other half of the secondary encoding.
        deviations: Standard deviation per block, drawn as a shaded band.
        cycles_per_block: Cycles held at each rate.
    """
    cycles, ys = step_series(values, cycles_per_block=cycles_per_block)
    if deviations is not None:
        _, sds = step_series(deviations, cycles_per_block=cycles_per_block)
        axis.fill_between(
            cycles, ys - sds, ys + sds, color=color, alpha=0.16, linewidth=0
        )
    axis.plot(
        cycles,
        ys,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=4.5,
        markerfacecolor=color if filled else SURFACE,
        markeredgecolor=color,
        markeredgewidth=1.2,
        label=label,
        zorder=3,
    )


def with_recovery(values: Sequence[float]) -> list[float]:
    """Append the return-to-0.1C recovery block to a tabulated rate series.

    Tables S8 and S9 stop at 20C; the corresponding figures then return to
    0.1C and show essentially full recovery for every material.  That final
    block is reconstructed here from the initial 0.1C value and is drawn and
    labelled as reconstructed wherever it appears.

    Args:
        values: The eight tabulated values.

    Returns:
        The nine block values including the recovery step.

    Examples:
        >>> with_recovery([1.0, 2.0])
        [1.0, 2.0, 1.0]
    """
    return [*list(values), values[0]]


RATES_WITH_RECOVERY = (*sd.RATE_TEST_C_RATES, 0.1)
