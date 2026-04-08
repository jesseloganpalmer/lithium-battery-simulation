"""This module defines the experiment to be simulated."""

from __future__ import annotations

import pybamm


def define_experiment() -> pybamm.Experiment:
    """Define the experiment to be simulated."""
    return pybamm.Experiment(
        [
            (
                "Discharge at C/10 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour",
            )
        ]
        * 3,
    )
