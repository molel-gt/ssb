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
             "cell energy density [Wh.kg-1]", "avg power density [W.kg-1]",
             "current density [A.m-2]", "discharge time [h]"]
L_sep = 50E-6

timestamp_now = datetime.utcnow().strftime("%Y-%m-%d-%H")


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

    # Study variables
    t_eval = np.linspace(0, 15 * 3600, 1000)
    cam_lengths = [50e-6, 100e-6, 200e-6, 300e-6, 400e-6, 600e-6, 1000e-6,
                   5000e-6]
    porosities = [0.4, 0.3, 0.2, 0.1]
    current_functions = [0.1e-3, 1e-3, 10e-3, 100e-3]

    #
    # Conduct study
    #
    #
    with open("studies/{}.csv".format(timestamp_now), "w") as fp:
        writer = csv.DictWriter(fp, fieldnames=col_names)
        writer.writeheader()
        for current_function in current_functions:
            params["Current function [A]"] = current_function
            for length in cam_lengths:
                for porosity in porosities:
                    file_name = "{length}_{porosity}_{current_density}".format(
                        length=str(int(length * 1e6)),
                        current_density=float(current_function * 1e4),
                        porosity=porosity)
                    params["Positive electrode thickness [m]"] = length
                    params["Positive electrode active material volume fraction"] = 1 - porosity
                    params["Positive electrode porosity"] = porosity
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
                    mass_cell = mass_res + rho_sse * (L_sep + porosity * length) + rho_cam * (1 - porosity) * length
                    energy = integrate.simps(sim.solution["Instantaneous power [W.m-2]"].data, sim.solution["Time [s]"].data) / 3600
                    avg_power = np.average(sim.solution["Instantaneous power [W.m-2]"].data) / np.average(sim.solution["Time [s]"].data / 3600)
                    t_d = max(sim.solution["Time [s]"].data) / 3600
                    row = {
                        "porosity": porosity, "separator length [m]": L_sep, "cathode length [m]": length,
                        "mass res [kg.m-2]": mass_res, "mass of cell [kg.m-2]": mass_cell,
                        "energy of cell [Wh.m-2]": energy, "cell energy density [Wh.kg-1]": energy / mass_cell,
                        "avg power density [W.kg-1]": avg_power / mass_cell,
                        "current density [A.m-2]": current_function * 1e4,
                        "discharge time [h]": t_d,
                    }
                    writer.writerow(row)

    # Ragone plots
    df = pd.read_csv("studies/" + timestamp_now + ".csv")

    porosities = [0.1, 0.2, 0.3, 0.4]
    cathode_lengths = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0006,
                       0.001]

    fig, axs = plt.subplots(2, 2)
    fig.suptitle('discharge time [h] vs cathode length [um]')

    for pos, current_density in enumerate(current_functions[-4:]):
        df2 = df[df["current density [A.m-2]"] == current_density * 1e4]
        x_pos = int(pos / 2)
        y_pos = pos % 2
        for porosity in porosities:
            data = df2[df2["porosity"] == porosity]
            x_data = data["cathode length [m]"]
            y_data = data["discharge time [h]"]
            axs[x_pos, y_pos].plot(x_data, y_data, linewidth=1,
                                   label="porosity: {}".format(porosity))

        axs[x_pos, y_pos].set_title("i = {} [A.m-2]".format(int(current_density * 1e4)))
        axs[x_pos, y_pos].legend()
        axs[x_pos, y_pos].tick_params(axis='y', which='both', direction='in', right=True)
        axs[x_pos, y_pos].set_box_aspect(1)
    plt.show()
