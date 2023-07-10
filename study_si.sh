#!/bin/bash

for type in lithiated delithiated; do
  sbatch study_si.sbatch $type
done