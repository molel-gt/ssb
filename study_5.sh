#!/bin/bash
echo 'Preparing Geometries..'

python3 contact_loss_2d.py --eps 0.1 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.2 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.3 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.4 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.6 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.7 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.8 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
python3 contact_loss_2d.py --eps 0.9 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --pos 0.25 --n_pieces 3
#
#echo 'Running Models..'
python3 current_constriction_2d.py --eps 0.1 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.2 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.3 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.4 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.6 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.7 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.8 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3
python3 current_constriction_2d.py --eps 0.9 --Lx 1 --Ly 1 --w 0.01 --h 0.1 --voltage 1 --pos 0.25 --n_pieces 3