#!/bin/bash
echo 'Creating Geometries..'
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.05
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.10
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.20
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.30
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.40
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.50
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.60
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.70
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.80
python3 contact_loss.py --grid_info 26-26-26 --contact_map data/current_constriction/test11.tif --eps 0.85

echo 'Running Models..'
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.05 --eps 0.05
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.1 --eps 0.10
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.2 --eps 0.20
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.3 --eps 0.30
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.4 --eps 0.40
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.5 --eps 0.50
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.6 --eps 0.60
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.7 --eps 0.70
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.8 --eps 0.80
python3 current_constriction.py --grid_size 25-25-25 --data_dir mesh/contact_loss/26-26-26/0.85 --eps 0.85