"""Calibrate the hierarchical LTO model against Sanders' measured rate tests.

The model is deliberately parameter-poor.  Almost everything it needs is read
straight out of the paper's tables.  Only three quantities are fitted, and each
is fitted against a different part of the data so that no single dataset both
sets and validates a parameter:

1. ``accessible_stoichiometry_span`` - one number, fixed by the measured 0.1C
   capacity of the reference electrode.  It carries no rate information.
2. ``lto_diffusivity_m2_s`` - one number, fitted globally against the two
   *droplet-templated* pure-LTO electrodes, whose lithium diffusion domain is
   known independently (it is the measured 86 nm primary crystallite radius).
3. ``diffusion_radius_m`` - fitted separately for every electrode, and this is
   the interesting one.  For templated microparticles it should come back at
   the primary radius; for the unstructured nanopowder it reveals how large
   its uncontrolled dry aggregates behave electrochemically.

Run with ``python Sanders_LTO/calibrate.py``.  Results are written to
``Sanders_LTO/results/``.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import time
from typing import Any

import lto_model as lm
import numpy as np
import sanders_data as sd
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)

#: Electrodes used to fit the global LTO diffusivity: droplet-templated, pure
#: LTO, so their diffusion domain is the measured primary crystallite.
DIFFUSIVITY_FIT_ELECTRODES = ("uniform", "polydisperse")

#: Electrode whose 0.1C capacity fixes the accessible stoichiometry span.
REFERENCE_ELECTRODE = "uniform"


def _rate_curve_rmse(
    structure: lm.Structure,
    calibration: lm.Calibration,
    measured: tuple[float, ...],
) -> float:
    """Root-mean-square capacity error over the eight rate-test points.

    Args:
        structure: Structural description of the electrode.
        calibration: Model calibration to evaluate.
        measured: Measured specific capacities [mAh/g] at
            :data:`sanders_data.RATE_TEST_C_RATES`.

    Returns:
        RMSE in mAh/g.
    """
    simulated = lm.rate_curve(structure, calibration)["specific_capacity_mah_g"]
    return float(np.sqrt(np.mean((simulated - np.asarray(measured)) ** 2)))


def fit_accessible_span(calibration: lm.Calibration) -> float:
    """Fix the accessible stoichiometry span from the measured 0.1C capacity.

    At 0.1C the cell is far from any transport limit, so the delivered
    capacity is set purely by how much of the theoretical 175 mAh/g lies
    inside the 1.0-2.5 V window.  That makes this parameter separable from
    everything else in the model.

    Args:
        calibration: Starting calibration (only the span is changed).

    Returns:
        The fitted accessible stoichiometry span.
    """
    target = sd.SPECIFIC_CAPACITY_MAH_G[REFERENCE_ELECTRODE][0]

    def cost(span: float) -> float:
        trial = dataclasses.replace(
            calibration, accessible_stoichiometry_span=float(span)
        )
        structure = lm.structure_from_electrode(REFERENCE_ELECTRODE, trial)
        capacity = lm.simulate_discharge(structure, trial, 0.1)[
            "specific_capacity_mah_g"
        ]
        return abs(capacity - target)

    result = minimize_scalar(
        cost, bounds=(0.85, 1.0), method="bounded", options={"xatol": 1e-4}
    )
    return float(result.x)


def fit_lto_diffusivity(calibration: lm.Calibration) -> float:
    """Fit the LTO solid-state diffusivity against the templated electrodes.

    Args:
        calibration: Starting calibration (only the diffusivity is changed).

    Returns:
        The fitted diffusivity in m2/s.
    """

    def cost(log10_diffusivity: float) -> float:
        trial = dataclasses.replace(
            calibration, lto_diffusivity_m2_s=10.0**log10_diffusivity
        )
        errors = []
        for key in DIFFUSIVITY_FIT_ELECTRODES:
            structure = lm.structure_from_electrode(key, trial)
            errors.append(
                _rate_curve_rmse(
                    structure, trial, sd.SPECIFIC_CAPACITY_MAH_G[key]
                )
            )
        return float(np.mean(errors))

    result = minimize_scalar(
        cost, bounds=(-18.0, -15.5), method="bounded", options={"xatol": 0.01}
    )
    return float(10.0**result.x)


def fit_diffusion_radius(
    key: str, calibration: lm.Calibration
) -> tuple[float, float]:
    """Fit one electrode's effective lithium diffusion domain radius.

    Args:
        key: Key into :data:`sanders_data.ELECTRODES`.
        calibration: Global calibration (diffusivity and span).

    Returns:
        A tuple ``(radius_m, rmse_mah_g)``.
    """
    measured = sd.SPECIFIC_CAPACITY_MAH_G[key]
    base = lm.structure_from_electrode(key, calibration)

    def cost(log10_radius: float) -> float:
        structure = dataclasses.replace(
            base, diffusion_radius_m=10.0**log10_radius
        )
        return _rate_curve_rmse(structure, calibration, measured)

    result = minimize_scalar(
        cost,
        bounds=(np.log10(4e-8), np.log10(3e-6)),
        method="bounded",
        options={"xatol": 0.01},
    )
    return float(10.0**result.x), float(result.fun)


def main() -> None:
    """Run the full calibration and write results to ``results/``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    calibration = lm.Calibration()

    logger.info("Fitting accessible stoichiometry span ...")
    span = fit_accessible_span(calibration)
    calibration = dataclasses.replace(
        calibration, accessible_stoichiometry_span=span
    )
    logger.info("  accessible stoichiometry span = %.4f", span)

    logger.info("Fitting LTO solid-state diffusivity ...")
    diffusivity = fit_lto_diffusivity(calibration)
    calibration = dataclasses.replace(
        calibration, lto_diffusivity_m2_s=diffusivity
    )
    logger.info("  D_LTO = %.3e m2/s", diffusivity)

    logger.info("Fitting per-electrode diffusion domain radii ...")
    rows: list[dict[str, Any]] = []
    for key, electrode in sd.ELECTRODES.items():
        radius_m, rmse = fit_diffusion_radius(key, calibration)
        primary_radius_m = sd.PRIMARY_PARTICLE_Z_AVERAGE_M / 2.0
        rows.append(
            {
                "electrode": key,
                "label": electrode.label,
                "secondary_diameter_um": electrode.secondary_diameter_um,
                "fitted_diffusion_radius_nm": radius_m * 1e9,
                "ratio_to_primary_radius": radius_m / primary_radius_m,
                "rmse_mah_g": rmse,
            }
        )
        logger.info(
            "  %-13s d_sec = %4.1f um -> R_diff = %6.1f nm (%.1fx primary),"
            " RMSE %.1f mAh/g",
            key,
            electrode.secondary_diameter_um,
            radius_m * 1e9,
            radius_m / primary_radius_m,
            rmse,
        )

    # The unstructured nanopowder's fitted radius is the model's headline
    # structure-sensitive parameter, so it is what gets stored.
    aggregate_radius_m = (
        next(
            r["fitted_diffusion_radius_nm"]
            for r in rows
            if r["electrode"] == "nano"
        )
        * 1e-9
    )
    calibration = dataclasses.replace(
        calibration, aggregate_radius_m=aggregate_radius_m
    )
    calibration.save()

    lm.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = lm.RESULTS_DIR / "fitted_diffusion_radii.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %s and %s", lm.CALIBRATION_FILE, out)
    logger.info("Calibration took %.0f s", time.monotonic() - started)


if __name__ == "__main__":
    main()
