import pybamm
from pybamm import exp, tanh

import matplotlib.pyplot as plt
import numpy as np


def open_circuit_potential_chen2020(sto):
    """
    LG M50 NMC open circuit potential as a function of stochiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stochiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open circuit potential
    """

    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq


def open_circuit_potential_unif(sto):
    """
    OCP which varies uniformly with reaction stoichiometry

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = pybamm.Scalar(4.7) - pybamm.Scalar(1.2) * sto

    return u_eq


def open_circuit_potential_min(sto):
    """
    OCP which varies uniformly with reaction stoichiometry

    Parameters
    ----------
    sto : :class:`pybamm.Symbol`
       Stochiometry of material (li-fraction)

    """

    u_eq = pybamm.Scalar(4.0) + pybamm.Scalar(1.2) * (sto - 0.5) ** 2

    return u_eq


def open_circuit_potential_sigmoid(sto):
    u_eq = 4.65 - 1 / (pybamm.exp(-3.5 * (sto - 0.5)) + 1)

    return u_eq


def open_circuit_potential(sto):
    return open_circuit_potential_chen2020(sto)


if __name__ == '__main__':
    stos = np.linspace(0.001, 0.99, 1000)
    ocp_chen2020_voltages = [open_circuit_potential_chen2020(sto).value for sto in stos]
    ocp_unif_voltages = [open_circuit_potential_unif(sto).value for sto in stos]
    ocp_min_voltages = [open_circuit_potential_min(sto).value for sto in stos]
    ocp_sigmoid_voltages = [open_circuit_potential_sigmoid(sto).value for sto in stos]
    fig, ax = plt.subplots()
    x_data = stos
    ax.plot(x_data, ocp_chen2020_voltages, label='NMC811 - Chen 2020')
    ax.plot(x_data, ocp_unif_voltages, label='U = 4.7 - 1.2 * sto')
    ax.plot(x_data, ocp_min_voltages, label='U = 4.0 + 1.2 * (sto - 0.5) ** 2')
    ax.plot(x_data, ocp_sigmoid_voltages, label='U = 4.3 - 1 / (exp(-3.5*(sto - 0.5)))')
    ax.set_xlabel("y")
    ax.set_ylabel("U [V]")
    ax.set_title("Open Circuit Potential (U) vs Insertion Fraction (y)")
    ax.legend()
    ax.grid()
    plt.savefig("open-circuit-potential.jpeg")
    plt.show()
