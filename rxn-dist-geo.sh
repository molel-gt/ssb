#!/bin/bash
for imgId in 6 11 16 22
do
python3 laminate_composite_positive_electrode_geo.py --dimensions 470-470-45 --lsep 15 --name_of_study reaction_distribution --img_id $imgId --radius 6 --eps_am 0.5 --scaling CONTACT_LOSS_SCALING --resolution 10 --refine
# mpiexec -n 8 python3 dg_charge_xfer.py --mesh_folder output/reaction_dist/470-470-45/$imgId/0.5/10.0/ --dimensions 470-470-45 --name_of_study reaction_dist  --Wa_p $Wa --gamma $gamma --kr $kr  #--atol 1e-15 --rtol 1e-14
done