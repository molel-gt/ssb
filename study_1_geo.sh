#!/bin/bash

for nx in 0 250;
do
  for ny in 0 250;
  do
    for nz in 0 200;
    do
      sbatch study_1_geo.sbatch 220-220-200_"$nx"-"$ny"-"$nz"
    done
  done
done