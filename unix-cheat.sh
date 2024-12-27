#!/bin/bash

# list files matching pattern and display results
find . -name \*.msh -print
find . -name \*unscaled* -print
# must install tree via sudo apt install tree
tree -if --noreport .
# split character
squeue -u emolel3 | awk '{split($0,a," "); print a[1]}' | xargs scancel

# list file in human-readable
ls -lh
ls -l --block-size=M
# slurmd slurmctld
sudo scontrol update nodename=localhost state=idle
sudo systemctl restart slurmctld && sudo systemctl restart slurmd && sudo systemctl restart slurmdbd && sudo systemctl restart munge && sinfo
sudo systemctl status slurmctld && sudo systemctl status slurmd

squeue -u molel | grep local | tr -s ' ' | cut -d ' ' -f 2 | xargs scancel

# find simulation files and print relevant lines
simfiles=$(find output/contact_loss_ref/ -name *.json | grep simulation)
for f in $simfiles;
do
    grep 'Dimensions' $f && grep 'Contact area fraction at left' $f && grep 'Max' $f;
done

# convert .ipynb to .py
jupyter nbconvert --to=python chapter-1-notebook.ipynb

# rename file
find output/contact_loss_lma/ -name trial.msh -exec bash -c 'echo mv $0 ${0/trial/mesh}' {} \;

# list disk usage
du -sh * | sort -h

# directory size
du -s ssb/ --block-size=MB

du -sh * | sort -hr

# install from requirements
conda install --file requirements.txt

# remove conda env
mamba remove --name paraview --all

# send output of find to copy

find output/reaction_distribution/75-40-0/ -name "*.png" | xargs cp -t figures/secondary_current

# stop docker from requiring sudo
sudo usermod -aG docker $USER
sudo setfacl -m user:$USER:rw /var/run/docker.sock

# docker share current directory
docker run -ti -v $(pwd):/root/shared -w /root/shared  dolfinx/dolfinx:nightly

# allow ownership of folder
sudo chown -R $(whoami):docker output

# building in apptainer/singularity
apptainer pull dolfinx.sif docker://dolfinx/dolfinx:nightly
singularity build dolfinx/dolfinx:nightly dolfinx.sif

# running program
module purge
module load gcc/10.3.0-o57x6h
module load openmpi/4.1.4
mpiexec -n 24 apptainer exec ~/dolfinx.sif python3

# copy files to cluster
scp src/ksp/ksp/impls/gmres/fgmres/fgmres.c emolel3@login-phoenix-rh9.pace.gatech.edu:/storage/coda1/p-tf74/0/shared/leshinka/softwares/petsc/src/ksp/ksp/impls/gmres/fgmres/

# fetch tags on forked branch
git fetch --tags https://github.com/petsc/petsc.git
git push --tags

# find resources in a partition
sinfo --partition=cpu-amd -o "%50N  %10c  %20m  %30G "

pace-check-queue cpu-amd
