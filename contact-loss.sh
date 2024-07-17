#!/usr/bin/env bash
if [ $# -eq 0 ] ; then
    echo 'usage: sh contact-loss.sh resolution'
    exit 1
fi

# mpiexec python3 contact_loss_separator.py --name_of_study contact_loss_lma --dimensions 470-470-1 --mesh_folder output/contact_loss_lma/470-470-1/11/0.5 --Wa 1000 --scaling CONTACT_LOSS_SCALING --no-compute_distribution --voltage 1
for img_id in 6 11 16 22
do
    for Lz in 1 15
    do
        python3 contact_loss_separator_lma_geo.py --img_id $img_id --dimensions 470-470-$Lz --resolution $1
    done
done