from __future__ import annotations

import pybamm

model1 = pybamm.lithium_ion.DFN()
model2 = pybamm.lithium_ion.DFN()

base_parameter_values = pybamm.ParameterValues("Chen2020")
updated_parameter_values = pybamm.ParameterValues("Chen2020")
experiment = pybamm.Experiment(["Discharge at 1C until 2.5 V"])

# 283K (~10°C) vs default 298.15K (~25°C) to 318K (~45°C)
updated_parameter_values["Ambient temperature [K]"] = 318.00
parameter_models = [
    pybamm.Simulation(
        model1, parameter_values=base_parameter_values, experiment=experiment
    ),
    pybamm.Simulation(
        model2, parameter_values=updated_parameter_values, experiment=experiment
    ),
]


sims = []

for sim in parameter_models:
    sim.solve()
    sims.append(sim)


pybamm.dynamic_plot(sims)
