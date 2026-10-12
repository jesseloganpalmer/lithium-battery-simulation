"""Joint sweep of both length scales: primary radius x secondary radius.

The two earlier sweeps each hold one scale fixed - :mod:`run_primary_sweep`
varies the crystallite inside Sanders' 8 um particle, :mod:`run_design_sweep`
varies the particle around Sanders' 172 nm powder.  This one varies both, to
answer the design question directly: which *pair* of sizes is best?

The map is computed twice, because the answer depends on one assumption that
the paper cannot constrain:

``independent``
    The two scales do not interact.  The primary sets the solid-state
    diffusion length and the interfacial area; the secondary sets electrode
    packing and intra-particle transport.  Nothing in the calibrated model
    couples them, so this is the model as fitted.

``carbon-limited``
    The secondary particle carries a fixed carbon *weight* fraction, so finer
    crystallites spread the same carbon over more internal surface and the
    intra-particle network gets thinner - see
    :func:`lto_model.carbon_limited_intra_particle_conductivity`.  This is a
    plausible but unmeasured coupling, and it is the only thing that can put a
    floor under how fine the primary should be.

Reading both panels together is the point: any conclusion that holds in one
and not the other is a consequence of that assumption, not of the data.

Run with ``python Sanders_LTO/run_scale_map.py``.
"""

from __future__ import annotations

import csv
import logging
import time

import lto_model as lm
import run_design_sweep as rds

logger = logging.getLogger(__name__)

#: Primary crystallite diameters [nm].  Sanders' size-selected powder is 172.
PRIMARY_DIAMETERS_NM = (30.0, 50.0, 80.0, 120.0, 172.0, 250.0, 400.0, 600.0)

#: Secondary particle diameters [um].  Sanders made 8.0.
SECONDARY_DIAMETERS_UM = (2.0, 4.0, 8.0, 12.0, 20.0, 35.0, 60.0)

#: Rates to map: one where the cell is comfortable, one where it is not.
MAP_C_RATES = (5.0, 20.0)

#: Areal loading, matching the coin cells of Figure 6.
LOADING_MG_CM2 = 1.22

COUPLING_MODES = ("independent", "carbon-limited")


def main() -> None:
    """Run the joint sweep and write ``results/scale_map.csv``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    calibration = lm.Calibration.load()
    correlations = rds.build_correlations()

    rows: list[dict[str, float | str]] = []
    for coupling in COUPLING_MODES:
        for primary_nm in PRIMARY_DIAMETERS_NM:
            primary_radius_m = primary_nm * 1e-9 / 2.0
            conductivity = (
                lm.carbon_limited_intra_particle_conductivity(primary_radius_m)
                if coupling == "carbon-limited"
                else lm.DEFAULT_INTRA_PARTICLE_CONDUCTIVITY_S_M
            )
            for secondary_um in SECONDARY_DIAMETERS_UM:
                # Compose the two tested pieces: the secondary size sets every
                # electrode-level property, then the primary is rescaled at
                # fixed intrinsic kinetics.
                structure = rds.design_structure(
                    diameter_um=secondary_um,
                    active_loading_mg_cm2=LOADING_MG_CM2,
                    intra_particle_conductivity_s_m=conductivity,
                    correlations=correlations,
                )
                structure = lm.rescale_primary_radius(
                    structure, primary_radius_m
                )
                for c_rate in MAP_C_RATES:
                    result = lm.simulate_discharge(
                        structure, calibration, c_rate
                    )
                    rows.append(
                        {
                            "coupling": coupling,
                            "primary_diameter_nm": primary_nm,
                            "secondary_diameter_um": secondary_um,
                            "c_rate": c_rate,
                            "intra_particle_conductivity_s_m": conductivity,
                            "electrode_thickness_um": (
                                structure.thickness_m * 1e6
                            ),
                            "fits_in_electrode": int(
                                secondary_um * 1e-6 <= structure.thickness_m
                            ),
                            "diffusion_time_constant_s": (
                                lm.diffusion_time_constant_s(
                                    primary_radius_m,
                                    calibration.lto_diffusivity_m2_s,
                                )
                            ),
                            **result,
                        }
                    )
            logger.info(
                "%s coupling, primary %.0f nm done", coupling, primary_nm
            )

    lm.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = lm.RESULTS_DIR / "scale_map.csv"
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
