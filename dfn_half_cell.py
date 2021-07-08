#!/usr/bin/env python3

import pybamm
import numpy as np
import matplotlib.pyplot as plt


options = {
    'working electrode': 'positive',
    }


def current_function(t):
    """
    Current pulse for 10 minutes followed by 10-minute relaxation
    with no current.
    """
    return 0.70e-3 * (t % 72000 <= 18000) - 0.35e-3 * (t % 72000 > 36000) * (t % 72000 <= 54000)


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Chen2020
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "1 + dlnf/dlnc": 1.0,
            "Cation transference number": 0.99,
            "Electrode height [m]": 1e-2,
            "Electrode width [m]": 1e-2,
            "Electrolyte diffusivity [m2.s-1]": 7.5e-12,
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            "Positive electrode active material volume fraction": 0.55,
            "Positive electrode diffusivity [m2.s-1]": 1e-13,
            'Positive electrode thickness [m]': 100e-06,
            "Positive electrode porosity": 0.45,
            "Positive particle radius [m]": 1e-6,
            "Separator porosity": 1.0,
            "Separator thickness [m]": 50e-6,
        },
        check_already_exists=False,
    )

    params["Initial concentration in negative electrode [mol.m-3]"] = 1000
    params["Current function [A]"] = current_function

    model = pybamm.lithium_ion.BasicDFNHalfCell(options=options)

    experiment = pybamm.Experiment(
        [
            (
                "Discharge at C/10 for 10 hours or until 3.5 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 50 mA",
                "Rest for 1 hour"
            ),
        ]
    )

    safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
    sim = pybamm.Simulation(model=model, parameter_values=params,
                            solver=safe_solver)
    t_eval = np.linspace(0, 3600 * 15, 1000)
    sim.solve(t_eval)

    sim.save("dfn-half-cell.pickle")

    sim.plot(
        [
            "Current density [A.m-2]",
            "Terminal voltage [V]",
            "Electrolyte concentration [mol.m-3]",
            [
                "Working electrode open circuit potential [V]",
                "Working electrode potential [V]",
            ],
            "Electrolyte potential [V]",
            "Specific power [W.m-2]",
            "Pore-wall flux [mol.m-2.s-1]",
            "Flux [mol.m-2.s-1]",
            # "Flux in electrolyte [mol.m-2.s-1]",
            # "Working particle surface concentration [mol.m-3]",
            "Working particle concentration [mol.m-3]",
            # "Current density divergence [A.m-3]",
        ],
        time_unit="hours",
        spatial_unit="um",
    )
