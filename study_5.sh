#!/bin/bash

for eps in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
do
  for pos in 0.1 0.5 0.9;
  do
    sbatch study_5.sbatch $eps $pos
  done
done