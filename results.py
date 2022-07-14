import csv

import matplotlib.pyplot as plt
import numpy as np


porosity = []
model = []
bruggeman = []

with open("results.csv", "r") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
        porosity.append(float(row['porosity']))
        model.append(float(row['model']))
        bruggeman.append(float(row['bruggeman']))
fig, ax = plt.subplots()
ax.plot(porosity, model, 'o')
ax.plot(porosity, bruggeman, '-')
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.1)
ax.set_box_aspect(1)
ax.set_xlabel(r"$\varepsilon$")
ax.set_ylabel(r"$\kappa_{eff}$")
ax.set_title(r"$\kappa_0$ = 0.1 S/m")
ax.legend(['model', 'bruggeman'])
plt.show()