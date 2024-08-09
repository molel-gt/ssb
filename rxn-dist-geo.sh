#!/bin/bash
Wa=0.1
gamma=15
kr=1
for imgId in 1 #6 11 16 22
do
    for eps in 0.1 #0.1 0.2 0.3 0.4 0.5 0.6 0.7
    do
    for se_pos_am_area_frac in 1.0 # 0.25 0.5 0.75 1.0
        do
            # python3 reaction_distribution_geo.py --name_of_study reaction_distribution --img_id $imgId --eps_am $eps --resolution 10 --se_pos_am_area_frac $se_pos_am_area_frac #--refine
            mpiexec -n 2 python3 dg_charge_xfer.py --mesh_folder output/reaction_distribution/470-470-45/15-30/$imgId/$eps/$se_pos_am_area_frac/unrefined/10.0/  --Wa_p $Wa --gamma $gamma --kr $kr  #--atol 1e-15 --rtol 1e-14
        done
    done
done
