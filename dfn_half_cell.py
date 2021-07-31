#!/usr/bin/env python3


import csv
import os

import pandas as pd
import pybamm
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from datetime import datetime


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
    "Instantaneous power [W.m-2]",
    "Electrolyte potential [V]",
    "Working particle surface concentration [mol.m-3]",
    "Electrolyte concentration [mol.m-3]",
    "Pore-wall flux [mol.m-2.s-1]",
]


# some densities
rho_cam = 2300  # NCM811 [kg.m-3]
rho_sse = 2254  # LSPS [kg.m-3]
mass_res = rho_sse * 50E-6  # residual mass of cell [kg.m-2]
col_names = ["porosity", "separator length [m]", "cathode length [m]",
             "mass res [kg.m-2]", "mass of cell [kg.m-2]", "energy of cell [Wh.m-2]",
             "specific energy [Wh.kg-1]", "specific power [W.kg-1]",
             "current density [A.m-2]", "discharge time [h]"]
L_sep = 50E-6

timestamp_now = datetime.utcnow().strftime("%Y-%m-%d-%H")

POROSITY = 0.3

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
            "Positive electrode active material volume fraction": 1 - POROSITY,
            # "Positive electrode conductivity [S.m-1]": 14,
            "Positive electrode diffusivity [m2.s-1]": 5e-13,
            # 'Positive electrode thickness [m]': 100e-6,
            "Positive electrode porosity": POROSITY,
            "Positive particle radius [m]": 1e-6,
            "Separator porosity": 1.0,
            "Separator thickness [m]": L_sep,
        },
        check_already_exists=False,
    )

    # Study variables
    t_max = 25 * 3600
    t_eval = np.linspace(0, t_max, 1000)
    cathode_lengths = [100e-6, 200e-6, 300e-6, 400e-6, 500e-6,
                       600e-6, 700e-6, 800e-6, 900e-6, 1000e-6]
    current_functions = [0.1e-3, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3,
                         1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3,
                         5e-3, 7.5e-3, 10e-3, 12.5e-3, 15e-3, 17.5e-3,
                         20e-3, 22.5e-3, 25e-3, 50e-3, 75e-3, 100e-3]

    #
    # Conduct study
    #

    with open("studies/{}.csv".format(timestamp_now), "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=col_names)
        writer.writeheader()
        for current_function in current_functions:
            params["Current function [A]"] = current_function
            for length in cathode_lengths:
                file_name = "{length}_{current_density}".format(
                    length=str(int(length * 1e6)),
                    current_density=float(current_function * 1e4))
                params["Positive electrode thickness [m]"] = length
                model = pybamm.lithium_ion.BasicDFNHalfCell(name=file_name, options=options)
                safe_solver = pybamm.CasadiSolver(atol=1e-3, rtol=1e-3, mode="safe")
                sim = pybamm.Simulation(model=model, parameter_values=params,
                                        solver=safe_solver)
                try:
                    sim.solve(t_eval)
                except Exception as e:
                    print(e)
                    continue
                sim.save(os.path.join("sims", file_name + ".pkl"))
                mass_cell = mass_res + rho_sse * (L_sep + POROSITY * length) + rho_cam * (1 - POROSITY) * length
                energy = integrate.simps(sim.solution["Instantaneous power [W.m-2]"].data, sim.solution["Time [s]"].data) / 3600
                avg_power = np.average(sim.solution["Instantaneous power [W.m-2]"].data) / np.average(sim.solution["Time [s]"].data / 3600)
                t_d = max(sim.solution["Time [s]"].data) / 3600
                row = {
                    "porosity": POROSITY, "separator length [m]": L_sep, "cathode length [m]": length,
                    "mass res [kg.m-2]": mass_res, "mass of cell [kg.m-2]": mass_cell,
                    "energy of cell [Wh.m-2]": energy, "specific energy [Wh.kg-1]": energy / mass_cell,
                    "specific power [W.kg-1]": avg_power / mass_cell,
                    "current density [A.m-2]": current_function * 1e4,
                    "discharge time [h]": t_d,
                }
                writer.writerow(row)

    # Visualize Results
    df = pd.read_csv("studies/" + timestamp_now + ".csv")
    df = df[df["porosity"] == POROSITY]

    fig1, ax1 = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    ax1.set_title('discharge time')
    for cat_len in cathode_lengths[:4]:
        df1 = df[df["cathode length [m]"] == cat_len]
        df1 = df1[df1["discharge time [h]"] < t_max / 3600]
        x_data = df1["current density [A.m-2]"]
        y_data = df1["discharge time [h]"]
        ax1.plot(x_data, y_data, linewidth=1, label="{} um".format(int(cat_len * 1e6)))
    ax1.legend()
    ax1.set_xlabel("current density [A.m-2]")
    ax1.set_ylabel("discharge time [h]")
    ax1.tick_params(axis='y', which='both', direction='in', right=True)
    ax1.set_box_aspect(1)
    plt.savefig("discharge-times.jpeg")
    plt.show()

    fig2, ax2 = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    ax2.set_title('ragone plot')
    for cat_len in cathode_lengths[:4]:
        df2 = df[df["cathode length [m]"] == cat_len]
        df2 = df2[df2["discharge time [h]"] < t_max / 3600]
        x_data = df2["specific power [W.kg-1]"]
        y_data = df2["specific energy [Wh.kg-1]"]
        ax2.plot(x_data, y_data, linewidth=1, label="{} um".format(int(cat_len * 1e6)))
    plt.axhline(y=500, color='grey', linestyle='--', linewidth=1, label='500 Wh/kg')
    plt.axvline(x=1000, color='grey', linestyle='-.', linewidth=1, label='1000 W/kg')
    ax2.legend()
    ax2.set_xlabel("Specific Power [W.kg-1]")
    ax2.set_ylabel("Specific Energy [Wh.kg-1]")
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    ax2.set_box_aspect(1)
    plt.savefig("ragone-plot.jpeg")
    plt.show()
