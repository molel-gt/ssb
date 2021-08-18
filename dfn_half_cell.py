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

# some densities
rho_cam = 2300  # NCM811 [kg.m-3]
rho_sse = 2254  # LSPS [kg.m-3]
mass_res = rho_sse * 50E-6  # residual mass of cell [kg.m-2]
col_names = ["porosity", "separator length [m]", "cathode length [m]",
             "mass res [kg.m-2]", "mass of cell [kg.m-2]", "energy of cell [Wh.m-2]",
             "specific energy [Wh.kg-1]", "specific power [W.kg-1]",
             "current density [A.m-2]", "discharge time [h]"]

output_vars = ["Terminal voltage [V]", "Working particle surface concentration",
                "Current density [A.m-2]", "Pore-wall flux [mol.m-2.s-1]",
                "Electrolyte concentration [mol.m-3]", "Electrolyte potential [V]",
                "Working electrode potential [V]",]

date_today = datetime.utcnow().strftime("%Y-%m-%d")


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Molel2021
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            "Separator thickness [m]": 25e-6,
        },
        check_already_exists=False,
    )

    porosity = params["Positive electrode porosity"]
    l_sep = params["Separator thickness [m]"]

    # Study variables
    t_max = 25 * 3600
    t_eval = np.linspace(0, t_max, 1000)
    cathode_lengths = [100e-6, 400e-6, 600e-6]
    current_functions = np.linspace(2.5e-3, 8.5e-3, 10)
    current_functions = [0.1e-3, 0.25e-3, 0.5e-3, 0.75e-3, 1e-3,
                         1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3,
                         5e-3, 7.5e-3, 10e-3, 12.5e-3, 15e-3, 17.5e-3,
                         20e-3, 22.5e-3, 25e-3, 50e-3, 75e-3, 100e-3, 1000e-3, 10000e-3]

    #
    # Conduct study
    #
    def get_var_permutations(var_a, var_b):
        """
        Return items based on loops from each var
        """
        for a in var_a:
            for b in var_b:
                yield a, b

    sims = []

    with open("studies/{}.csv".format(date_today), "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=col_names)
        writer.writeheader()
        for current_function, length in get_var_permutations(current_functions, cathode_lengths):
            params["Current function [A]"] = current_function
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
            sims.append(sim)
            sim.save(os.path.join("sims", file_name + ".pkl"))
            mass_cell = mass_res + rho_sse * (l_sep + porosity * length) + rho_cam * (1 - porosity) * length
            energy = integrate.simps(sim.solution["Instantaneous power [W.m-2]"].data, sim.solution["Time [s]"].data) / 3600
            avg_power = np.average(sim.solution["Instantaneous power [W.m-2]"].data) / np.average(sim.solution["Time [s]"].data / 3600)
            t_d = max(sim.solution["Time [s]"].data) / 3600
            row = {
                "porosity": porosity, "separator length [m]": l_sep, "cathode length [m]": length,
                "mass res [kg.m-2]": mass_res, "mass of cell [kg.m-2]": mass_cell,
                "energy of cell [Wh.m-2]": energy, "specific energy [Wh.kg-1]": energy / mass_cell,
                "specific power [W.kg-1]": avg_power / mass_cell,
                "current density [A.m-2]": current_function * 1e4,
                "discharge time [h]": t_d,
            }
            writer.writerow(row)

    pybamm.dynamic_plot(sims[-5], output_vars)

    # Visualize Results
    df = pd.read_csv("studies/" + date_today + ".csv")

    fig1, ax1 = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    ax1.set_title('discharge capacity')
    for cat_len in cathode_lengths:
        df1 = df[df["cathode length [m]"] == cat_len]
        df1 = df1[df1["discharge time [h]"] < t_max / 3600]
        x_data = df1["current density [A.m-2]"]
        y_data = df1["discharge time [h]"]
        ax1.scatter(x_data, x_data * y_data, linewidth=1, label="{} um".format(int(cat_len * 1e6)))
    ax1.legend()
    ax1.set_xlabel("current density [A.m-2]")
    ax1.set_ylabel("discharge capacity [Ah.m-2]")
    ax1.tick_params(axis='y', which='both', direction='in', right=True)
    ax1.set_box_aspect(1)
    ax1.grid()
    plt.savefig("ocp-chen2020-discharge-capacity.jpeg")
    plt.show()

    fig2, ax2 = plt.subplots()
    plt.xscale('log')
    plt.yscale('log')
    ax2.set_title('ragone plot')
    for cat_len in cathode_lengths:
        df2 = df[df["cathode length [m]"] == cat_len]
        df2 = df2[df2["discharge time [h]"] < t_max / 3600]
        x_data = df2["specific power [W.kg-1]"]
        y_data = df2["specific energy [Wh.kg-1]"]
        ax2.scatter(x_data, y_data, linewidth=1, label="{} um".format(int(cat_len * 1e6)))
    ax2.legend()
    ax2.set_xlabel("Specific Power [W.kg-1]")
    ax2.set_ylabel("Specific Energy [Wh.kg-1]")
    ax2.tick_params(axis='y', which='both', direction='in', right=True)
    ax2.set_box_aspect(1)
    ax2.grid()
    plt.savefig("ocp-chen2020-ragone-plot.jpeg")
    plt.show()
