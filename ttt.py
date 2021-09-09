#!/usr/bin/env python3

import pybamm
import numpy as np


pybamm.set_logging_level("INFO")

options = {
    # 'working electrode': 'both',
}

output_vars = ["Terminal voltage [V]",  # "Current density [A.m-2]",
               "Pore-wall flux [mol.m-2.s-1]", "Electrolyte concentration [mol.m-3]",
               "Electrolyte potential [V]",
               [
                   "Working electrode potential [V]", "Working electrode open circuit potential [V]",
               ],
               "Working particle surface concentration [mol.m-3]"]


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Molel2021
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
        },
        check_already_exists=False,
    )

    experiment = pybamm.Experiment(
        [
            ("Discharge at C/5 for 5 hours or until 3.5 V",
             "Rest for 1 hour",
             "Charge at 1 A until 4.1 V",
             "Hold at 4.1 V until 50 mA",
             "Rest for 1 hour"),
        ] * 2
    )

    t_max = 1 * 3600
    t_eval = np.linspace(0, t_max, 1000)

    model = pybamm.lithium_metal.DFN(name='dfn-assb', options=options)
    safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
    fast_solver = pybamm.CasadiSolver(mode="fast")
    sim = pybamm.Simulation(model, parameter_values=params, experiment=experiment, solver=fast_solver)
    sim.solve()

    sim.plot(time_unit="hours")
