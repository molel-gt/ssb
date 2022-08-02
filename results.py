import csv

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


axis_font = {'size': '24'}
porosity = []
model = []
bruggeman = []
kappa = 0.1
with open("results.csv", "r") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
        porosity.append(float(row['porosity']))
        model.append(float(row['model']))
        bruggeman.append(float(row['bruggeman']))
laminate = [kappa * v for v in porosity]
fig, ax = plt.subplots()
ax.plot(porosity, model, 'o')
ax.plot(porosity, bruggeman, '-')
ax.plot(porosity, kappa * np.array(porosity) ** 2.5, '-')
ax.plot(porosity, laminate, '-')
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.1)
ax.set_box_aspect(1)
ax.set_xlabel(r"$\varepsilon$", **axis_font)
ax.set_ylabel(r"$\kappa_{eff}$", **axis_font)
ax.set_title(r"$\kappa_0$ = 0.1 S/m", **axis_font)
ax.legend(['FEniCSx model', r'Bruggeman model: $\kappa \varepsilon^{1.5}$', r'$\kappa \varepsilon^{2.5}$', r'Laminate: $\kappa \varepsilon$'], prop={"size": "14"})
plt.show()