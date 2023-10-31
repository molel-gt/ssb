#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "requires L_z and password"
    exit 1
fi
for img_id in 6 11 16 22;
do
  sshpass -p $2 scp emolel3@login-phoenix-slurm.pace.gatech.edu:/storage/coda1/p-tf74/0/shared/leshinka/ssb/mesh/study_2/test$img_id/470-470-$1_000-000-000/frequency.csv data/frequency-$img_id-$1.csv
done