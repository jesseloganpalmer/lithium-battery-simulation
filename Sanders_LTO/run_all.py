"""Run the whole study end to end.

Order matters: the calibration writes the fitted parameters that the forward
sweep and the model figures read back.

Run with ``python Sanders_LTO/run_all.py`` (about four minutes).
"""

from __future__ import annotations

import logging
import time

import calibrate
import plot_calibration
import plot_design_map
import plot_experimental
import plot_primary_sweep
import plot_scale_map
import run_design_sweep
import run_primary_sweep
import run_scale_map

logger = logging.getLogger(__name__)

STEPS = (
    ("Replot Sanders' measured rate tests", plot_experimental.main),
    ("Calibrate the model against Tables S8/S10", calibrate.main),
    ("Compare the calibrated model with experiment", plot_calibration.main),
    ("Sweep secondary particle size forward", run_design_sweep.main),
    ("Draw the design map", plot_design_map.main),
    ("Sweep primary crystallite size at 8 um", run_primary_sweep.main),
    ("Draw the primary-size figure", plot_primary_sweep.main),
    ("Map both length scales jointly", run_scale_map.main),
    ("Draw the joint design map", plot_scale_map.main),
)


def main() -> None:
    """Run every step in order."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    started = time.monotonic()
    for index, (description, step) in enumerate(STEPS, start=1):
        logger.info("[%d/%d] %s", index, len(STEPS), description)
        step()
    logger.info("all steps finished in %.0f s", time.monotonic() - started)


if __name__ == "__main__":
    main()
