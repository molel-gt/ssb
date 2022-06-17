import csv

import matplotlib.pyplot as plt
import numpy as np


porosity = []
kappa_eff = []
bruggeman = []

with open("results.csv", "r") as fp:
    reader = csv.DictReader(fp)
    for row in reader:
        porosity.append(float(row['porosity (avg)']))
        kappa_eff.append(float(row['kappa_eff (avg)']))
        bruggeman.append(float(row['bruggeman']))
fig, ax = plt.subplots()
ax.plot(porosity, kappa_eff, 'o')
ax.plot(porosity, bruggeman, '--')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_box_aspect(1)
ax.axis('equal')
ax.legend(['model avg @cc', 'bruggeman'])
plt.show()