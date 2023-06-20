#!/bin/bash

for imgId in 0..202;
do
  python3 edge_detection.py --img_id $imgId
done