#!/bin/bash

for img_id in 1 6 7 10 11 13 16 22;
do
  sbatch study_2_geo.sbatch $img_id
done