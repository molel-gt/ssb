#!/bin/bash

cd $HPC_DIR

if [ -d $HPC_DIR/spack ]; then
    echo 'directory exists, skip cloning'
    git stash && git pull origin
else
    git clone https://github.com/spack/spack.git
fi

cd $HPC_DIR/spack
# git checkout v0.23.0
sed -i 's#$tempdir/$user/spack-stage#/storage/coda1/p-tf74/0/shared/leshinka/spack-temp#g' ./etc/spack/defaults/config.yaml
. $HPC_DIR/spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack external find slurm
spack external find openmpi
spack add petsc^strumpack~slate
spack add adios2
spack add fenics-dolfinx@main+adios2+petsc^strumpack~slate py-fenics-dolfinx@main cflags="-O3" fflags="-O3"
spack add py-gmsh
spack install
spack load py-pip
python3 -m pip install matplotlib scipy
spack add py-gmsh
spack install
