#!/usr/bin/env python3


import os

import pybamm
import numpy as np


pybamm.set_logging_level("INFO")

options = {
    'working electrode': 'positive',
}

current_function = 2e-3

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

output_variables = [
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
]


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Chen2020
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "1 + dlnf/dlnc": 1.0,
            "Cation transference number": 1,
            "Electrode height [m]": 1e-2,
            "Electrode width [m]": 1e-2,
            # "Electrolyte diffusivity [m2.s-1]": 7.5e-12,
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            # "Positive electrode active material volume fraction": 0.55,
            # "Positive electrode diffusivity [m2.s-1]": 1e-13,
            'Positive electrode thickness [m]': 100e-6,
            # "Positive electrode porosity": 0.45,
            # "Positive particle radius [m]": 1e-6,
            "Separator porosity": 1.0,
            "Separator thickness [m]": 50e-6,
        },
        check_already_exists=False,
    )

    params["Initial concentration in negative electrode [mol.m-3]"] = 1000
    params["Current function [A]"] = current_function

    # Study variables
    t_eval = np.linspace(0, 20000, 1000)
    cam_lengths = [100e-6, 200e-6, 300e-6, 400e-6]
    cam_vol_fracs = [0.2, 0.3, 0.5, 0.7, 0.8]

    #
    # Conduct study
    #
    all_sim_files = []
    for length in cam_lengths:
        for cam_vol_frac in cam_vol_fracs:
            file_name = "L{}PHI{}".format(str(int(length * 1e6)),
                                              str(cam_vol_frac).replace(".", ""))
            if cam_vol_frac != "":
                params["Positive electrode thickness [m]"] = length
                params["Positive electrode active material volume fraction"] = cam_vol_frac
                params["Positive electrode porosity"] = 1 - cam_vol_frac
            model = pybamm.lithium_ion.BasicDFNHalfCell(name=file_name, options=options)
            safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
            sim = pybamm.Simulation(model=model, parameter_values=params,
                                    solver=safe_solver)
            sim.solve(t_eval)
            sim.save(file_name + ".pkl")

    sim_files = [f for f in os.listdir(".") if f.startswith("L3") and f.endswith(".pkl")]

    sims = []
    for sim_file in sim_files:
        sim = pybamm.load(sim_file)
        sims.append(sim)
    pybamm.dynamic_plot(sims, output_variables=output_variables,
                        time_unit="seconds", spatial_unit="um")
