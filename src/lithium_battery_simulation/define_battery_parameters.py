"""This module defines the experiment to be simulated."""

from __future__ import annotations

import pybamm


def define_battery_parameters() -> pybamm.Experiment:
    """Define the battery parameters to be simulated."""
    parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values["Electrode height [m]"] = 0.1  # example value
    parameter_values["Negative particle radius [m]"] = 1e-6
    parameter_values["Positive particle radius [m]"] = 1e-3
    return parameter_values


if __name__ == "__main__":
    parameter_values = pybamm.ParameterValues("Chen2020")
