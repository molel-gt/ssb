import os


def make_dir_if_missing(f_path):
    """"""
    os.makedirs(f_path, exist_ok=True)


def print_dict(dict, padding=20):
    for k, v in dict.items():
        print(k.ljust(padding, ' '), ": ", str(v))


def nmc_capacity(density, volume, Ni=0.6, Mn=0.2, Co=0.2):
    """
    density of nmc [kg/m3]

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
