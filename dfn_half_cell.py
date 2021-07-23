#!/usr/bin/env python3


import csv
import os

import pybamm
import numpy as np
import scipy.integrate as integrate


pybamm.set_logging_level("INFO")

options = {
    'working electrode': 'positive',
}

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
    "Instantaneous power [W.m-2]",
    "Pore-wall flux [mol.m-2.s-1]",
    "Working particle surface concentration [mol.m-3]",
]


# some densities
rho_cam = 2300  # NCM811 [kg.m-3]
rho_sse = 2254  # LSPS [kg.m-3]
mass_res = rho_sse * 50E-6  # residual mass of cell that is not cathode or separator [kg.m-2] 
col_names = ["porosity", "sep length [m]", "cat length [m]", "mass res [kg.m-2]",
            "mass of cell [kg.m-2]", "energy of cell [Wh.m-2]", "cell energy density [Wh.kg-1]"]
L_sep = 50E-6


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
            "Electrolyte conductivity [S.m-1]": 1.0,
            "Electrolyte diffusivity [m2.s-1]": 5e-12,
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            # "Positive electrode active material volume fraction": 0.55,
            # "Positive electrode conductivity [S.m-1]": 14,
            "Positive electrode diffusivity [m2.s-1]": 5e-13,
            # 'Positive electrode thickness [m]': 100e-6,
            # "Positive electrode porosity": 0.45,
            "Positive particle radius [m]": 1e-6,
            "Separator porosity": 1.0,
            "Separator thickness [m]": L_sep,
        },
        check_already_exists=False,
    )

    # params["Initial concentration in negative electrode [mol.m-3]"] = 1000
    params["Current function [A]"] = 10e-3

    # Study variables
    t_eval = np.linspace(0, 36000, 1000)
    cam_lengths = [100e-6, 200e-6, 300e-6, 400e-6, 600e-6, 1000e-6, 5000e-6]
    cam_vol_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9]

    #
    # Conduct study
    #

    with open("study.csv", "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=col_names)
        writer.writeheader()
        for length in cam_lengths:
            for cam_vol_frac in cam_vol_fracs:
                file_name = "L{}PHI{}".format(str(int(length * 1e6)),
                                              str(cam_vol_frac).replace(".", ""))
                params["Positive electrode thickness [m]"] = length
                params["Positive electrode active material volume fraction"] = cam_vol_frac
                params["Positive electrode porosity"] = 1 - cam_vol_frac
                model = pybamm.lithium_ion.BasicDFNHalfCell(name=file_name, options=options)
                safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
                sim = pybamm.Simulation(model=model, parameter_values=params,
                                        solver=safe_solver)
                sim.solve(t_eval)
                sim.save(file_name + ".pkl")
                mass_cell = mass_res + rho_sse * (L_sep + (1 - cam_vol_frac) * length) + rho_cam * cam_vol_frac * length
                energy = integrate.simps(sim.solution["Instantaneous power [W.m-2]"].data, sim.solution["Time [s]"].data) / 3600
                row = {
                    "porosity": 1 - cam_vol_frac, "sep length [m]": L_sep, "cat length [m]": length,
                    "mass res [kg.m-2]": mass_res, "mass of cell [kg.m-2]": mass_cell, 
                    "energy of cell [Wh.m-2]": energy, "cell energy density [Wh.kg-1]": energy / mass_cell
                }
                writer.writerow(row)

    select_sims = []
    sim_files = [f for f in os.listdir(".") if any([f.startswith("L2"), f.startswith("L3"), f.startswith("L4"), f.startswith("L6")]) and f.endswith("8.pkl")]
    for sim_file in sim_files:
        sim = pybamm.load(sim_file)
        select_sims.append(sim)
    pybamm.dynamic_plot(select_sims, output_variables=output_variables,
                        time_unit="hours", spatial_unit="um")
