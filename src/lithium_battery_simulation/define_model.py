"""Defines the lithium battery model to be simulated."""

from __future__ import annotations

import pybamm


def define_model() -> pybamm.BaseModel:
    """Define the lithium battery model to be simulated."""
    return pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model


if __name__ == "__main__":
    model = pybamm.lithium_ion.DFN(options={"particle size": "distribution"})
    model.print_parameter_info()
