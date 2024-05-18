import os

import numpy as np
import ufl


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


def print_dict(dict, padding=20):
    for k, v in dict.items():
        print(k.ljust(padding, ' '), ": ", str(v))


def compute_angles_in_triangle(p1, p2, p3):
    B = np.array(p3) - np.array(p1)
    C = np.array(p2) - np.array(p1)
    A = np.array(p2) - np.array(p3)
    a = np.arccos(np.dot(B, C) / (np.linalg.norm(B) * np.linalg.norm(C)))
    b = np.arccos(np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C)))
    c = np.arccos(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))

    return a, b, c


def nmc_capacity(density, volume, Ni=0.6, Mn=0.2, Co=0.2):
    """
    :density: [kg/m3]
    :volume:  m3

    return: capacity in [A.h]
    """
    faraday_constant = 96485  # A.s/mol
    mwt_li = 6.94e-3  # kg/mol
    mwt_ni = 58.693e-3  # kg/mol
    mwt_mn = 54.938e-3  # kg/mol
    mwt_co = 58.933e-3  # kg/mol
    mwt_o = 15.999e-3  # kg/mol
    mass_frac_li = mwt_li / (mwt_li + Ni * mwt_ni + Mn * mwt_mn + Co * mwt_co + 2 * mwt_o)
    moles_li = density * volume * mass_frac_li / mwt_li

    return moles_li * faraday_constant * (1 / 3600)


def c_rate_current(capacity, c_rate=1):
    """
    capacity [A.h]

    returns `current` [A]
    """
    return capacity * c_rate


def lithium_concentration_nmc(density, Ni=0.6, Mn=0.2, Co=0.2):
    """
    :density: [kg/m3]
    :volume:  m3

    :returns: concentration in mol/m3
    """
    mwt_li = 6.94e-3  # kg/mol
    mwt_ni = 58.693e-3  # kg/mol
    mwt_mn = 54.938e-3  # kg/mol
    mwt_co = 58.933e-3  # kg/mol
    mwt_o = 15.999e-3  # kg/mol
    mass_frac_li = mwt_li / (mwt_li + Ni * mwt_ni + Mn * mwt_mn + Co * mwt_co + 2 * mwt_o)

    return density * mass_frac_li / mwt_li


def arcsinh(x):
    return ufl.ln(x + ufl.sqrt(x ** 2 + 1))
