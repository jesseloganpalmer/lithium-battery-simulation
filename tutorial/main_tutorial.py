from __future__ import annotations

import pybamm as pb

models = [pb.lithium_ion.SPM(), pb.lithium_ion.SPMe(), pb.lithium_ion.DFN()]

sims = []
for model in models:
    sim = pb.Simulation(model)
    sim.solve([0, 3600])
    sims.append(sim)


pb.dynamic_plot(sims)
