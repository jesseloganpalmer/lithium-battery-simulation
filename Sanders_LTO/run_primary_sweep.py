"""Sweep the primary crystallite size inside a fixed 8 um secondary particle.

The calibration in :mod:`calibrate` established that a droplet-templated
secondary particle diffuses over its *primary* crystallites: three templated
samples spanning 2.3 to 9.1 um all returned the same 85-89 nm diffusion
domain.  That makes the primary radius, not the secondary diameter, the length
that sets the rate limit - and it is a length the templating route can change
independently, simply by emulsifying a different precursor nanopowder.

Sanders et al. use one size-selected powder (172 nm Z-average) throughout, so
this sweep is a prediction rather than a comparison.  It holds the secondary
particle at the 8.0 um they actually made and asks what a different precursor
would buy.

What is held fixed, and why:

* electrode packing (thickness, porosity, density) - it is set by the
  secondary particle, which is not changing.  This is the paper's own point:
  the secondary particle protects the packing density while the primary sets
  the rate.
* intra-particle porosity - random packing of monodisperse spheres has a
  size-independent void fraction, so shrinking the primaries changes the pore
  *size*, not the pore *fraction*.
* the intrinsic exchange-current density - see
  :func:`lto_model.rescale_primary_radius` for why the measured
  charge-transfer resistance has to be rescaled to achieve this.

Run with ``python Sanders_LTO/run_primary_sweep.py``.
"""

from __future__ import annotations

import csv
import logging
import time

import lto_model as lm
import sanders_data as sd

logger = logging.getLogger(__name__)

#: Primary crystallite diameters to explore [nm].  Sanders' powder is 172 nm.
PRIMARY_DIAMETERS_NM = (
    20.0,
    30.0,
    50.0,
    80.0,
    120.0,
    172.0,
    250.0,
    400.0,
    600.0,
    1000.0,
    1500.0,
)

#: The secondary particle is held at the size Sanders actually made.
FIXED_SECONDARY_DIAMETER_UM = 8.0

#: Above roughly this primary size the dispersed phase stops being "more than
#: an order of magnitude smaller than the membrane pores" (5 um), which the
#: Supporting Information flags as the point where blockages become a risk.
MEMBRANE_BLOCKING_PRIMARY_NM = 500.0


def main() -> None:
    """Run the primary-size sweep and write ``results/primary_sweep.csv``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    calibration = lm.Calibration.load()

    # Start from the measured membrane-emulsified electrode, so every
    # electrode-level property is the one Sanders reported for it.
    base = lm.structure_from_electrode("uniform", calibration)

    rows: list[dict[str, float]] = []
    for diameter_nm in PRIMARY_DIAMETERS_NM:
        structure = lm.rescale_primary_radius(base, diameter_nm * 1e-9 / 2.0)
        tau_s = lm.diffusion_time_constant_s(
            structure.diffusion_radius_m, calibration.lto_diffusivity_m2_s
        )
        for c_rate in sd.RATE_TEST_C_RATES:
            result = lm.simulate_discharge(structure, calibration, c_rate)
            rows.append(
                {
                    "primary_diameter_nm": diameter_nm,
                    "secondary_diameter_um": FIXED_SECONDARY_DIAMETER_UM,
                    "c_rate": c_rate,
                    "diffusion_time_constant_s": tau_s,
                    "discharge_duration_s": 3600.0 / c_rate,
                    "charge_transfer_resistance_ohm": (
                        structure.charge_transfer_resistance_ohm
                    ),
                    **result,
                }
            )
        logger.info(
            "primary %.0f nm: tau = %.0f s, 20C capacity = %.1f mAh/g",
            diameter_nm,
            tau_s,
            rows[-1]["specific_capacity_mah_g"],
        )

    lm.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = lm.RESULTS_DIR / "primary_sweep.csv"
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
