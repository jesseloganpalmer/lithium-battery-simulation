"""Run the simulation."""  # noqa: INP001

from __future__ import annotations

import pybamm

from lithium_battery_simulation.define_battery_parameters import (
    define_battery_parameters,
)
from lithium_battery_simulation.define_experiment import define_experiment
from lithium_battery_simulation.define_model import define_model
from lithium_battery_simulation.define_timeframe import define_timeframe


def run_simulation() -> None:
    """Run the lithium battery simulation."""
    experiment = define_experiment()
    parameter_values = define_battery_parameters()
    model = define_model()
    time_frame = define_timeframe()

    sim = pybamm.Simulation(
        model, parameter_values=parameter_values, experiment=experiment
    )
    sim.solve(time_frame)  # solve for the defined timeframe
    sim.plot()


if __name__ == "__main__":
    run_simulation()
