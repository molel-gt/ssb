#!/bin/bash
echo 'Preparing Geometries..'

python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 1
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 2
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 3
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 4
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 5
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 6
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 7
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 8
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 9
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 10

echo 'Running Models..'
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 1
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 2
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 3
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 4
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 5
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 6
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 7
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 8
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 9
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 10