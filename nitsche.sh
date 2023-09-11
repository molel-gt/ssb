#!/bin/bash
#if [ "$#" -ne 1 ]; then
#    echo "requires L_z"
#    exit 1
#fi
python3 nitsche_geo.py --root_folder mesh/nitsche_diffusivity/61-51-0_000-000-000 --resolution 0.1
python3 nitsche_diffusivity.py --root_folder mesh/nitsche_diffusivity --grid_extents 61-51-0_000-000-000