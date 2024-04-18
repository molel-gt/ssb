#!/usr/bin/env bash

mpiexec python3 contact_loss_separator.py --name_of_study contact_loss_lma --dimensions 470-470-1 --mesh_folder output/contact_loss_lma/470-470-1/11/0.5 --Wa 1000 --scaling CONTACT_LOSS_SCALING --no-compute_distribution --voltage 1