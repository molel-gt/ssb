#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/spack ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/spack/spack.git
    . ./spack/share/spack/setup-env.sh
    spack env create fenicsx-env
    spack env activate fenicsx-env
    spack add fenics-dolfinx+adios2 py-fenics-dolfinx cflags="-O3" fflags="-O3"
    spack install
fi