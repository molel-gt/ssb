#!/usr/bin/env bash

# echo "Started on `/bin/hostname`"
# for w in 10 50 90; do
#     for h in 0.25 0.50 0.75; do
#         python3 multiple_domains.py --Lx 100 --Ly 1 --w $w --h $h
#         for cov in 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 1.0; do
#             echo "Running model for coverage " $cov
#             mpirun -n 4 python3 current_constriction.py --coverage $cov --Lx 100 --Ly 1 --w $w --h $h --voltage 1
#             mpirun -n 4 python3 current_constriction.py --coverage $cov --Lx 100 --Ly 1 --w $w --h $h --voltage 2
#         done
#     done
# done

echo "Started on `/bin/hostname`"

for cov in 0.25; do
    for pos in 'mid'; do
        for num_piece in 1 2 5 10 50 75 100 250 350 500; do
            python3 multiple_domains.py --Lx 100 --Ly 1 --pos $pos --n_pieces $num_piece --eps $cov
            echo "running model for piece: " $num_piece
            mpirun -n 1 python3 current_constriction.py --eps $cov --Lx 100 --Ly 1 --voltage 1 --pos $pos --n_pieces $num_piece
            mpirun -n 1 python3 current_constriction.py --eps $cov --Lx 100 --Ly 1 --voltage 2 --pos $pos --n_pieces $num_piece
        done
    done
done
./summary.py