#!/usr/bin/env python3

import pybamm


pybamm.set_logging_level("INFO")


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Molel2021
    params = pybamm.ParameterValues(chemistry=chemistry)

    experiment = pybamm.Experiment(
        [
            (
             "Discharge at C/100 for 20 hours or until 3.0 V",
             "Rest for 2 hours",
             "Charge at C/100 until 4.1 V",
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

    # plot
    sim.plot(
        [
            "Total current density [A.m-2]",
            "Terminal voltage [V]",
            "Positive electrode potential [V]",
            "Negative electrode potential [V]",
            "Electrolyte potential [V]",
            "Positive electrode open circuit potential [V]",
            "Electrolyte concentration [mol.m-3]",
        ],
        time_unit="seconds", spatial_unit="um")
