#!/bin/bash
for relative_radius in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9;
do
  for Wa in 0.00001 0.0001 0.001 0.01 0.1 1 10 100 1000;
  do
    python3 study_3_geo.py --relative_radius $relative_radius --outdir mesh/study_3/$Wa/$relative_radius && python3 study_3.py --outdir mesh/study_3/$Wa/$relative_radius --Wa $Wa
  done
done