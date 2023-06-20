#!/bin/bash

# list files matching pattern and display results
find . -name \*.msh -print
find . -name \*unscaled* -print
# must install tree via sudo apt install tree
tree -if --noreport .
# split character
squeue -u emolel3 | awk '{split($0,a," "); print a[1]}' | xargs scancel

# list file in human-readable
ls -lh
ls -l --block-size=M
# slurmd slurmctld
sudo scontrol update nodename=localhost state=idle
sudo systemctl restart slurmctld && sudo systemctl restart slurmd && sudo systemctl restart slurmdbd
sudo systemctl status slurmctld && sudo systemctl status slurmd

squeue -u molel | grep local | tr -s ' ' | cut -d ' ' -f 2 | xargs scancel