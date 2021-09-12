#!/usr/bin/env python3

import pybamm


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Xu2019
    params = pybamm.ParameterValues(chemistry=chemistry)

    experiment = pybamm.Experiment(
        [
            (
             "Discharge at C/5 for 5 hours or until 3.0 V",
             "Rest for 3 hours",
             "Charge at 1mA until 4.1 V",
             "Hold at 4.1 V until 50 mA",
             "Rest for 3 hours"
             ),
        ] * 2
    )

    model = pybamm.lithium_metal.DFN()
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-3)
    sim = pybamm.Simulation(model, parameter_values=params,
                            experiment=experiment, solver=solver)

    sim.solve()
    sim.plot(time_unit="seconds")
