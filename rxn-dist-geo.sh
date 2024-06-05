#!/bin/bash

python3 laminate_composite_positive_electrode_geo.py --dimensions 470-470-45 --lsep 15 --name_of_study reaction_distribution --img_id 11 --radius 6 --eps_am 0.5 --scaling CONTACT_LOSS_SCALING --resolution 3
mpiexec python3 lmb_3d_cc_new.py --mesh_folder output/reaction_distribution/470-470-45/11/0.5/3.0/ --dimensions 470-470-45 --name_of_study reaction_distribution  --Wa_p $Wa --gamma $gamma --kr $kr  #--atol 1e-15 --rtol 1e-14
