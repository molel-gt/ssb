#!/usr/bin/env python3

import pybamm
import matplotlib.pyplot as plt
import numpy as np


T = pybamm.Parameter("Temperature [K]")
R = pybamm.Parameter("Molar gas constant [J.mol-1.K-1]")
F = pybamm.Parameter("Faraday constant [C.mol-1]")

options = {
    'operating mode': 'current',
    'dimensionality': 0,
    'surface form': 'false',
    'convection': 'none',
    'current collector': 'uniform',
    'particle': 'Fickian diffusion',
    'particle shape': 'spherical',
    'electrolyte conductivity': 'default',
    'thermal': 'isothermal',
    'cell geometry': 'arbitrary',
    'external submodels': [], 'SEI': 'none',
    'lithium plating': 'none',
    'SEI porosity change': 'false',
    'lithium plating porosity change': 'false',
    'loss of active material': 'none',
    'working electrode': 'positive',
    'particle mechanics': 'none',
    'total interfacial current density as a state': 'false',
    'SEI film resistance': 'none'
    }


def cation_transference_number(c_e, T):
    return 0.0107907 + 1.48837e-4 * c_e


def open_circuit_potential(c_s_p):
    return 2.7 + (R * T / F) * (-0.000558 * c_s_p + 8.10)


def current_function(t):
    """
    Current pulse for 10 minutes followed by 10-minute relaxation with no current.
    """
    return 0.25 * (t % 1200 <= 600)


if __name__ == '__main__':
    # default parameters
    chemistry = pybamm.parameter_sets.Chen2020
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "Cation transference number": cation_transference_number,
            "Discharge capacity [A.h]": 50,
            "EC diffusivity [m2.s-1]": 7.5e-12,
            "EC initial concentration in electrolyte [mol.m-3]": 1000,
            "Electrode cross-sectional area [m2]": 1e-4,
            "Electrolyte diffusivity [m2.s-1]": 7.5e-12,
            "Faraday constant [C.mol-1]": 96485,
            "Initial concentration in electrolyte [mol.m-3]": 1000,
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            "Lower voltage cut-off [V]": 1.7,
            "Maximum concentration in the solid [mol.m-3]": 29000,
            "Molar gas constant [J.mol-1.K-1]": 8.314,
            "Positive electrode active material volume fraction": 0.65,
            "Positive electrode diffusivity [m2.s-1]": 5e-13,
            "Positive electrode OCP [V]": open_circuit_potential,
            'Positive electrode thickness [m]': 100e-06,
            "Positive electrode porosity": 30e-2,
            "Positive particle radius [m]": 1e-6,
            "Separator porosity": 0.3,
            "Separator thickness [m]": 50e-6,
            "Temperature [K]": 373.15,
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
    sim = pybamm.Simulation(model=model, parameter_values=params)
    sim.solve([0, 900])
    sim.plot(
        [
            "Working particle concentration [mol.m-3]",
            "Electrolyte concentration [mol.m-3]",
            "Current [A]",
            "Working electrode potential [V]",
            "Electrolyte potential [V]",
            "Total electrolyte concentration",
            "Total lithium in working electrode [mol]",
            "Working electrode open circuit potential [V]",
            "Lithium counter electrode exchange-current density [A.m-2]",
            ["Terminal voltage [V]", "Voltage drop in the cell [V]"],
        ]
    )
