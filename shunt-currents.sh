#!/bin/bash

for L_p in 0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05
do
    python3 shunt_currents.py --mesh_folder shunt-current --L_p $L_p --vary L_p
done

for N_s in 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100
do
    echo 'N_s: '$N_s
    python3 shunt_currents.py --mesh_folder shunt-current --N_s $N_s --vary N_s
done

for w in 10 20 50 100 200 500 1000 2000 5000 10000
do
    echo 'w: '$w
    python3 shunt_currents.py --mesh_folder shunt-current --w $w --vary w
done
