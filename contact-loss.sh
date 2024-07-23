#!/usr/bin/env bash
if [ $# -eq 0 ] ; then
    echo 'usage: sh contact-loss.sh resolution'
    exit 1
fi

for img_id in 22 #6 11 16 22
do
    for Lz in 1 5 15 30 50
    do
        # python3 contact_loss_separator_lma_geo.py --img_id $img_id --dimensions 470-470-$Lz --resolution $1
        mpiexec -n 24 python3 contact_loss_separator_amg.py --mesh_folder output/contact_loss_lma/470-470-$Lz/$img_id/unrefined/$1/
    done
done
