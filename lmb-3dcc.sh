#!/bin/bash

resolution=0.001

# python3 two_subdomains_2d.py --name_of_study secondary_current --resolution $resolution #--refine
# python3 lmb_3d_cc_2d_geo.py --name_of_study tertiary_current --resolution $resolution --refine
# python3 secondary_current_geo.py --name_of_study ssb_complex --resolution $resolution #--refine
# python3 secondary_current_2d_geo.py --name_of_study tertiary_current --dimensions 75-40-0 --resolution $resolution #--refine

# for Wa in 1e3
# do
# 	echo 'Running for Wa = '$Wa
# 	for kr in 0.01 # 1 100
# 	do
# 		echo 'Running for gamma = '$gamma
# 		# rm -r ~/.cache/fenics
# 		# mpiexec -n 2 python3 lmb_3d_cc_new.py --name_of_study lmb_planar --dimensions 150-40-0 --mesh_folder output/ssb_planar/150-40-0/20-55-20/5.0e-06/ --gamma $gamma --Wa_p $Wa
# 		rm -r ~/.cache/fenics
# 		mpiexec -n 1 python3 secondary_current.py --mesh_folder output/reaction_distribution/75-40-0/$resolution --gamma $gamma --Wa_p $Wa --kr $kr --atol 1e-15 --rtol 1e-12 --kinetics butler_volmer --plot
# 		# rm -r ~/.cache/fenics
# 		# mpiexec -n 2 python3 secondary_current.py --mesh_folder output/ssb_complex/40-40-75/unrefined/$resolution --gamma $gamma --Wa_p $Wa --kr $kr --atol 1e-15 --rtol 1e-12 --kinetics butler_volmer
# 	done
# done
# rm -r ~/.cache/fenics
# python3 tertiary_current_hdg.py --mesh_folder output/tertiary_current/150-40-0/20-55-20/5.0e-06/ --kr 1

# mpiexec -n 3 python3 secondary_current_cg.py --mesh_folder output/tertiary_current/75-40-0/unrefined/1.0/ --Wa_p 1e3 --gamma 100 --kr 10
# mpiexec -n 3 python3 secondary_current_cg.py --mesh_folder output/secondary_current/150-40-0/20-55-20/5.0/ --Wa_p 1e3 --gamma 100 --kr 10

# mpiexec -n 1 python3 secondary_current_cg.py --mesh_folder output/secondary_current/1-0-0/ --Wa_p 0.1 --gamma 15 --kr 1 --kinetics linear
# mpiexec -n 1 python3 tertiary_current_cg.py --mesh_folder output/secondary_current/1-0-0/ --Wa_p 1 --gamma 15 --kr 1 --kinetics butler_volmer --voltage 4.2
# mpiexec -n 1 python3 tertiary_current.py --mesh_folder output/secondary_current/100-0-0/ --Wa_p 1 --gamma 15 --kr 1 --kinetics butler_volmer --voltage 4.2
# mpiexec -n 1 python3 tertiary_current.py --mesh_folder output/tertiary_current/150-40-0/$resolution --Wa_p 1 --gamma 15 --kr 1 --kinetics butler_volmer --voltage 4.2
mpiexec -n 1 python3 tertiary_current.py --mesh_folder output/reaction_distribution/40-40-75/unrefined/0.01 --Wa_p 1 --gamma 15 --kr 1 --kinetics butler_volmer --voltage 4.2

# mpiexec -n 1 python3 secondary_current_cg.py --mesh_folder output/tertiary_current/75-40-0/unrefined/$resolution/ --Wa_p 0.1 --gamma 15 --kr 1 --voltage 4.2 #--kinetics linear
# mpiexec -n 1 python3 tertiary_current_cg.py --mesh_folder output/tertiary_current/75-40-0/unrefined/$resolution/ --Wa_p 1 --gamma 15 --kr 1 --voltage 4.2 --kinetics butler_volmer

# mpiexec -n 1 python3 tertiary_current_cg.py --mesh_folder output/secondary_current/150-40-0/20-55-20/$resolution/ --Wa_p 1 --gamma 15 --kr 1 --voltage 4.2
# mpiexec -n 1 python3 tertiary_current_cg.py --mesh_folder output/tertiary_current/40-40-75/unrefined/1/ --Wa_p 1 --gamma 150 --kr 1 --voltage 4.2
