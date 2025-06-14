#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/spack ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/spack/spack.git
    cd $SOFTWARES_DIR/spack
    # git checkout v0.23.0
    cp $WORK_DIR/../spack/config.yaml $SOFTWARES_DIR/spack/
    . $SOFTWARES_DIR/spack/share/spack/setup-env.sh
    spack env create fenicsx-env
    spack env activate fenicsx-env
    spack add fenics-dolfinx@main%gcc@12.3.0+adios2+petsc py-fenics-dolfinx%gcc@12.3.0 cflags="-O3" fflags="-O3"
    spack add py-gmsh
    spack install
fi
