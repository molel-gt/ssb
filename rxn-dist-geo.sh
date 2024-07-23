#!/bin/bash
for imgId in 6 11 16 22
do
    for eps in 0.7 #0.1 0.2 0.3 0.4 0.5 0.6 0.7
    do
        python3 laminate_composite_positive_electrode_geo.py --name_of_study reaction_distribution --img_id $imgId --eps_am $eps --resolution 10 --refine
        # mpiexec -n 8 python3 dg_charge_xfer.py --mesh_folder output/reaction_distribution/470-470-45/$imgId/15-30/0.5/10.0/ --dimensions 470-470-45 --name_of_study reaction_dist  --Wa_p $Wa --gamma $gamma --kr $kr  #--atol 1e-15 --rtol 1e-14
    done
done
