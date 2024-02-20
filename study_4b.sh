#!/bin/bash

for Lz in 1 5 15 30 50 100;
  do
  for eps in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75;
    do
      sbatch study_4b.sbatch $eps 470-470-$Lz
    done
 done
