#!/bin/bash
echo 'Creating Geometries..'
for Lz in 7.25 26;
do
#  for eps in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75;
  for eps in 0.01 0.05 0.1 0.15; # 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75;
  do
    python3 study_4_geo.py --eps $eps --grid_extents 26-26-$Lz
  done
done
