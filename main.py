import pybamm
from pybamm.parameters import parameter_values

import params


model = pybamm.lithium_ion.DFN()
parameter_values = pybamm.ParameterValues(values=params.parameters)
sim = pybamm.Simulation(model=model, parameter_values=parameter_values)
sim.solve([0, 3600])
sim.plot()