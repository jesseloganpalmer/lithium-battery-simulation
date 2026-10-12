"""Forward sweep: what does secondary particle size actually buy you?

The calibration in :mod:`calibrate` fixes the model against Sanders' three
measured LTO electrodes.  This script then uses it *predictively*, over a range
of secondary particle diameters the paper did not make, to answer the design
question the paper raises but cannot fully answer experimentally: how large
should a droplet-templated secondary particle be?

Two competing effects set the answer, and both are in the model:

* Larger, more uniform secondary particles pack into denser electrodes
  (Table S6) and lower the contact and charge-transfer resistances
  (Tables S10/S11).  Both correlations are fitted to Sanders' measured points
  and extrapolated with a saturating form.
* Inside a secondary particle, electrons must cross the intra-particle carbon
  network and Li+ the intra-particle electrolyte to reach every primary
  nanocrystal.  That drop grows as the square of the secondary radius, so
  beyond some size the particle starts to starve its own interior.

The intra-particle electronic conductivity is the least well constrained input
in the whole model - the paper measures electrode-level sheet resistance, not
the conductivity inside a single microparticle - so it is swept over two
decades rather than asserted.

Run with ``python Sanders_LTO/run_design_sweep.py``.
"""

from __future__ import annotations

import csv
import dataclasses
import logging
import time

import lto_model as lm
import numpy as np
import sanders_data as sd

logger = logging.getLogger(__name__)

#: Secondary particle diameters to explore [um].  Sanders reach 8-9 um; the
#: sweep runs an order of magnitude beyond that to find where it stops paying.
DESIGN_DIAMETERS_UM = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0)

#: Intra-particle carbon-network conductivities to bracket [S/m].  The middle
#: value is the module default for a ~2.2 wt % pyrolysed-carbon skeleton; the
#: upper value stands in for the CNT-loaded composite microparticles.
INTRA_PARTICLE_CONDUCTIVITIES_S_M = (1e-4, 1e-3, 1e-2)

#: Areal active-material loadings to run [mg/cm2]: the thin coin-cell
#: electrodes of Figure 6, and the thick electrode of Figure S14.
LOADINGS_MG_CM2 = (1.22, 15.5)

#: Density of the carbon-black plus PVDF fraction of the coating [kg/m3].
CARBON_BINDER_DENSITY_KG_M3 = 1800.0

#: Electrodes whose measured properties anchor the size correlations.
CORRELATION_KEYS = ("nano", "polydisperse", "uniform")


def build_correlations() -> dict[str, tuple[float, float, float]]:
    """Fit the measured electrode-property correlations against size.

    Returns:
        A mapping from property name to a
        :func:`lto_model.fit_saturating_property` model.
    """
    diameters = [
        sd.ELECTRODES[k].secondary_diameter_um for k in CORRELATION_KEYS
    ]
    return {
        "electrode_density_g_cm3": lm.fit_saturating_property(
            diameters,
            [
                sd.ELECTRODES[k].electrode_density_g_cm3
                for k in CORRELATION_KEYS
            ],
        ),
        "series_resistance_ohm": lm.fit_saturating_property(
            diameters,
            [
                float(sd.ELECTRODES[k].eis_r0_ohm or 0.0)
                + float(sd.ELECTRODES[k].eis_r1_ohm or 0.0)
                for k in CORRELATION_KEYS
            ],
        ),
        "charge_transfer_resistance_ohm": lm.fit_saturating_property(
            diameters,
            [
                float(sd.ELECTRODES[k].eis_r3_ohm or 0.0)
                for k in CORRELATION_KEYS
            ],
        ),
    }


def design_structure(
    *,
    diameter_um: float,
    active_loading_mg_cm2: float,
    intra_particle_conductivity_s_m: float,
    correlations: dict[str, tuple[float, float, float]],
) -> lm.Structure:
    """Build the structure of a hypothetical droplet-templated electrode.

    Electrode thickness follows from the areal loading and the size-dependent
    packing density; porosity follows from the solid volume actually present.

    Args:
        diameter_um: Secondary particle diameter.
        active_loading_mg_cm2: Active-material areal loading.
        intra_particle_conductivity_s_m: Conductivity of the carbon network
            inside a secondary particle.
        correlations: Fitted size correlations from :func:`build_correlations`.

    Returns:
        The structure to simulate.
    """
    density_g_cm3 = lm.evaluate_saturating_property(
        diameter_um, correlations["electrode_density_g_cm3"]
    )
    areal_loading_mg_cm2 = active_loading_mg_cm2 / sd.ACTIVE_MASS_FRACTION
    # thickness [m] = areal loading [kg/m2] / electrode density [kg/m3]
    thickness_m = (areal_loading_mg_cm2 * 1e-2) / (density_g_cm3 * 1e3)

    active_kg_m2 = active_loading_mg_cm2 * 1e-2
    carbon_binder_kg_m2 = (areal_loading_mg_cm2 - active_loading_mg_cm2) * 1e-2
    solid_volume_m = (
        active_kg_m2 / sd.LTO_CRYSTAL_DENSITY_KG_M3
        + carbon_binder_kg_m2 / CARBON_BINDER_DENSITY_KG_M3
    )
    porosity = float(np.clip(1.0 - solid_volume_m / thickness_m, 0.05, 0.95))

    return lm.Structure(
        primary_radius_m=sd.PRIMARY_PARTICLE_Z_AVERAGE_M / 2.0,
        # A droplet-templated particle is porous and electrolyte filled, so
        # the diffusion domain stays the primary crystallite - which is what
        # the calibration independently recovered for every templated sample.
        diffusion_radius_m=sd.PRIMARY_PARTICLE_Z_AVERAGE_M / 2.0,
        secondary_diameter_m=diameter_um * 1e-6,
        intra_particle_porosity=lm.DEFAULT_INTRA_PARTICLE_POROSITY,
        intra_particle_conductivity_s_m=intra_particle_conductivity_s_m,
        thickness_m=thickness_m,
        active_loading_kg_m2=active_kg_m2,
        porosity=porosity,
        series_resistance_ohm=lm.evaluate_saturating_property(
            diameter_um, correlations["series_resistance_ohm"]
        ),
        charge_transfer_resistance_ohm=lm.evaluate_saturating_property(
            diameter_um, correlations["charge_transfer_resistance_ohm"]
        ),
    )


def unstructured_reference(
    *,
    active_loading_mg_cm2: float,
    calibration: lm.Calibration,
) -> lm.Structure:
    """The unstructured-nanopowder baseline at a given loading.

    Args:
        active_loading_mg_cm2: Active-material areal loading.
        calibration: Fitted model quantities.

    Returns:
        The structure of an unstructured electrode at that loading.
    """
    base = lm.structure_from_electrode("nano", calibration)
    scale = (active_loading_mg_cm2 * 1e-2) / base.active_loading_kg_m2
    return dataclasses.replace(
        base,
        active_loading_kg_m2=active_loading_mg_cm2 * 1e-2,
        thickness_m=base.thickness_m * scale,
    )


def main() -> None:
    """Run the design sweep and write ``results/design_sweep.csv``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    calibration = lm.Calibration.load()
    correlations = build_correlations()
    for name, model in correlations.items():
        logger.info(
            "correlation %-32s v0=%.3g v_inf=%.3g d_scale=%.1f um",
            name,
            *model,
        )

    rows: list[dict[str, float | str]] = []

    for loading in LOADINGS_MG_CM2:
        baseline = unstructured_reference(
            active_loading_mg_cm2=loading, calibration=calibration
        )
        for c_rate in sd.RATE_TEST_C_RATES:
            result = lm.simulate_discharge(baseline, calibration, c_rate)
            rows.append(
                {
                    "structure": "unstructured",
                    "diameter_um": 0.0,
                    "loading_mg_cm2": loading,
                    "intra_particle_conductivity_s_m": float("nan"),
                    "c_rate": c_rate,
                    "electrode_thickness_um": baseline.thickness_m * 1e6,
                    "electrode_porosity": baseline.porosity,
                    "fits_in_electrode": 1,
                    **result,
                }
            )
        logger.info("baseline done for %.2f mg/cm2", loading)

        for conductivity in INTRA_PARTICLE_CONDUCTIVITIES_S_M:
            for diameter in DESIGN_DIAMETERS_UM:
                structure = design_structure(
                    diameter_um=diameter,
                    active_loading_mg_cm2=loading,
                    intra_particle_conductivity_s_m=conductivity,
                    correlations=correlations,
                )
                for c_rate in sd.RATE_TEST_C_RATES:
                    result = lm.simulate_discharge(
                        structure, calibration, c_rate
                    )
                    rows.append(
                        {
                            "structure": "templated",
                            "diameter_um": diameter,
                            "loading_mg_cm2": loading,
                            "intra_particle_conductivity_s_m": conductivity,
                            "c_rate": c_rate,
                            "electrode_thickness_um": (
                                structure.thickness_m * 1e6
                            ),
                            "electrode_porosity": structure.porosity,
                            # A secondary particle cannot be thicker than the
                            # coating that contains it; points that violate
                            # this are kept but flagged, not silently plotted.
                            "fits_in_electrode": int(
                                diameter * 1e-6 <= structure.thickness_m
                            ),
                            **result,
                        }
                    )
                logger.info(
                    "loading %.2f mg/cm2, sigma %.0e S/m, d = %.0f um done",
                    loading,
                    conductivity,
                    diameter,
                )

    lm.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = lm.RESULTS_DIR / "design_sweep.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(
        "wrote %s (%d rows) in %.0f s",
        out,
        len(rows),
        time.monotonic() - started,
    )


if __name__ == "__main__":
    main()
