import math
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pybamm

import utils as ssb_utils

params = ssb_utils.load_all_params()
parameters = pybamm.ParameterValues(values=params)
model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model=model, parameter_values=parameters)
sim.solve([0, 3600])
sim.plot()

Molel2021 = {
    "chemistry": "lithium-ion",
    "cell": "LGM50_Chen2020",
    "negative electrode": "graphite_Chen2020", # update to lithium_Molel2021
    "separator": "separator_Chen2020",  # dimensions should be set to 0
    "positive electrode": "nmc_Chen2020",
    "electrolyte": "lipf6_Nyman2008",
    "experiment": "1C_discharge_from_full_Chen2020",  # update to 1C_discharge_from_full_Molel2021
    "sei": "example",
    "citation": "Chen2020",  # update to (Molel2021)
}