#!/bin/bash

for eps in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
do
  for pos in 0.1 0.5 0.9;
  do
    python3 contact_loss_2d.py --eps $eps --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos $pos --n_pieces 3 && python3 current_constriction_2d.py --eps $eps --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos $pos --n_pieces 3
  done
done

#python3 contact_loss_2d.py --eps 0.1 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.2 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.3 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.4 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.6 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.7 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.8 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 contact_loss_2d.py --eps 0.9 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#
#echo 'Running Models..'
#python3 current_constriction_2d.py --eps 0.1 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.2 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.3 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.4 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.5 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.6 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.7 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.8 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3
#python3 current_constriction_2d.py --eps 0.9 --Lx 1 --Ly 1 --w 0.1 --h 0.1 --pos 0.5 --n_pieces 3