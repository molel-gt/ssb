#!/usr/bin/env python3

import pybamm


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


def open_circuit_potential(c_s_surf_w):
    return 2.7 + (R * T / F) * (-0.000558 * c_s_surf_w + 8.10)


def current_function(t):
    """
    Current pulse for 10 minutes followed by 10-minute relaxation
    with no current.
    """
    return 0.7e-3 * (t % 14400 <= 3600) - 0.7e-3 * (t % 14400 >= 7200) * (t % 14400 <= 10800)


if __name__ == '__main__':

    # default parameters
    chemistry = pybamm.parameter_sets.Chen2020
    params = pybamm.ParameterValues(chemistry=chemistry)

    params.update(
        {
            "1 + dlnf/dlnc": 1.0,
            "Cation transference number": 0.75,
            "Discharge capacity [A.h]": 5,
            "Electrode height [m]": 1e-2,
            "Electrode width [m]": 1e-2,
            "Electrolyte diffusivity [m2.s-1]": 7.5e-12,
            "Electrolyte conductivity [S.m-1]": 0.18,
            "Faraday constant [C.mol-1]": 96485,
            "Initial concentration in electrolyte [mol.m-3]": 1500,
            "Lithium counter electrode exchange-current density [A.m-2]": 12.6,
            "Lithium counter electrode conductivity [S.m-1]": 1.0776e7,
            "Lithium counter electrode thickness [m]": 50e-6,
            "Maximum concentration in positive electrode [mol.m-3]": 23720,
            "Molar gas constant [J.mol-1.K-1]": 8.314,
            "Positive electrode active material volume fraction": 0.675,
            "Positive electrode conductivity [S.m-1]": 1e3,
            "Positive electrode diffusivity [m2.s-1]": 1e-13,
            "Positive electrode exchange-current density [A.m-2]": 13.1,
            'Positive electrode thickness [m]': 100e-06,
            "Positive electrode porosity": 0.4,
            "Positive particle radius [m]": 2e-6,
            "Separator porosity": 1.0,
            "Separator thickness [m]": 50e-6,
            "Temperature [K]": 353.15,
            "Lower cut-off voltage [V]": 1.1,
            "Upper cut-off voltage [V]": 4.3,
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
    sim.solve([0, 3600 * 30])
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
            # "Flux in electrolyte [mol.m-2.s-1]",
            "Total electrolyte concentration [mol]",
            "Total lithium in working electrode [mol]",
            # "Working particle surface concentration [mol.m-3]",
            # "Working particle concentration [mol.m-3]",
            # "Current density divergence [A.m-3]",
        ],
        time_unit="hours",
        spatial_unit="um",
    )
