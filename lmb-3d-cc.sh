#!/usr/bin/env bash

for gamma in 3 5; do
    sbatch lmb-3d-cc.sbatch $gamma
done