#!/bin/bash
echo 'Creating Geometries..'
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.05
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.10
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.20
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.30
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.40
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.50
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.60
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.70
#python3 study_4_geo.py --grid_info 26-26-7.25 --contact_map data/current_constriction/test11.tif --eps 0.75

echo 'Running Models..'
#python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.05 --eps 0.05
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.1 --eps 0.10
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.2 --eps 0.20
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.3 --eps 0.30
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.4 --eps 0.40
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.5 --eps 0.50
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.6 --eps 0.60
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.7 --eps 0.70
python3 study_4.py --grid_size 25-25-25 --data_dir mesh/study_4/26-26-7.25/0.75 --eps 0.75
