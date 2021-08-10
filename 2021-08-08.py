#!/usr/bin/env python3


import csv
import os

import pandas as pd
import pybamm
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate


pybamm.set_logging_level("INFO")

options = {
    'working electrode': 'positive',
}

# some densities
rho_cam = 2300  # NCM811 [kg.m-3]
rho_sse = 2254  # LSPS [kg.m-3]
mass_res = rho_sse * 50E-6  # residual mass of cell [kg.m-2]
col_names = ["porosity", "separator length [m]", "cathode length [m]",
             "mass res [kg.m-2]", "mass of cell [kg.m-2]", "energy of cell [Wh.m-2]",
             "specific energy [Wh.kg-1]", "specific power [W.kg-1]",
             "current density [A.m-2]", "discharge time [h]"]
L_SEP = 50E-6

output_vars = ["Terminal voltage [V]",
                "Current density [A.m-2]", "Pore-wall flux [mol.m-2.s-1]",
                "Electrolyte concentration [mol.m-3]", "Electrolyte potential [V]",
                ["Working electrode potential [V]", "Working electrode open circuit potential [V]", ],
                "Working particle surface concentration [mol.m-3]"]

POROSITY = 0.3


def current_function(t):
    return 0.70e-3 * (t % 72000 <= 18000) - 0.35e-3 * (t % 72000 > 36000) * (t % 72000 <= 54000)


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

    # Study variables
    t_max = 20 * 3600
    t_eval = np.linspace(0, t_max, 1000)

    current_functions = np.linspace(0.1e-3, 100e-3, 25)
    current_functions = [0.1e-3, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3,
                         1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3,
                         5e-3, 7.5e-3, 10e-3, 12.5e-3, 15e-3, 17.5e-3,
                         20e-3, 22.5e-3, 25e-3, 50e-3, 75e-3, 100e-3]

    cat_length = 100E-6
    params["Current function [A]"] = current_function
    params["Positive electrode thickness [m]"] = cat_length
    model = pybamm.lithium_ion.BasicDFNHalfCell(name=cat_length, options=options)
    safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
    sim = pybamm.Simulation(model=model, parameter_values=params,
                            solver=safe_solver)
    sim.solve(t_eval)
    sim.save(str(cat_length) + ".pkl")
    mass_cell = mass_res + rho_sse * (L_SEP + POROSITY * cat_length) + rho_cam * (1 - POROSITY) * cat_length
    energy = integrate.simps(sim.solution["Instantaneous power [W.m-2]"].data, sim.solution["Time [s]"].data) / 3600
    avg_power = np.average(sim.solution["Instantaneous power [W.m-2]"].data) / np.average(sim.solution["Time [s]"].data / 3600)
    t_d = max(sim.solution["Time [s]"].data) / 3600

    pybamm.dynamic_plot(sim, output_vars, time_unit="hours")
    # fig, ax = plt.subplots()
    # ax.plot(sim.solution["Working particle surface concentration [mol.m-3]"].data,
    #         sim.solution["Working electrode potential [V]"].data[:, 0])
    # plt.show()
