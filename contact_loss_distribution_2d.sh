#!/bin/bash
python3 contact_loss_2d.py --eps 1 --Lx 10 --Ly 10 --w 0 --h 0 --pos 0 --n_pieces 1

python3 current_constriction_2d.py --eps 1 --Lx 10 --Ly 10 --w 0 --h 0 --voltage 1 --pos 0 --n_pieces 1
python3 current_constriction_2d.py --eps 1 --Lx 10 --Ly 10 --w 0 --h 0 --voltage 2 --pos 0 --n_pieces 1