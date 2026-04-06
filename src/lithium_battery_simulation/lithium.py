"""Lithium battery simulation example."""

from __future__ import annotations

import pybamm

experiment = pybamm.Experiment(
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

model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
sim = pybamm.Simulation(model, experiment=experiment)
sim.solve([0, 10 * 3600])  # solve for 10 hours
sim.plot()
