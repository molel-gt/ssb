#!/bin/bash

module load anaconda3 && conda activate fenicsx-env

for imgId in `seq 0 201`;
do
  python3 edge_detection.py --img_id $imgId
done