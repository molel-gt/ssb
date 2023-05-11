#!/bin/bash
echo 'Preparing Geometries..'
python3 contact_loss_2d.py --eps 1 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 1

python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 1
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 10
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 50
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 100
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 150
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 200
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 250
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 300
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 350
python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --pos 0 --n_pieces 400
echo 'Running Models..'
python3 current_constriction_2d.py --eps 1 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 1
python3 current_constriction_2d.py --eps 1 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 1

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 1
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 1

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 10
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 10

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 50
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 50

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 100
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 100

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 150
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 150

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 200
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 200

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 250
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 250

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 300
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 300

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 350
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 350

python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 400
python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 400