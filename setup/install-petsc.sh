#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/petsc ]; then
    echo 'Directory exists, not cloning'
    git stash && git checkout main && git pull origin
else
    git clone https://gitlab.com/petsc/petsc.git
fi

cd $SOFTWARES_DIR/petsc
git checkout v3.23.2

./configure --with-clean --download-f2cblaslapack COPTFLAGS='-O2' CXXOPTFLAGS='-O2' FOPTFLAGS='-O2' --with-mpi-dir=$CMAKE_INSTALL_PREFIX --download-strumpack --download-hypre --with-64-bit-indices=no --download-metis --download-parmetis --download-ptscotch --download-eigen --download-hdf5 PETSC_ARCH=$PETSC_ARCH --with-scalar-type=real --with-shared-libraries --with-debugging=no --download-superlu_dist --download-scalapack --with-log=1 --with-cuda=0 --use-gpu-aware-mpi=0 --download-netcdf --download-zlib --download-ml --download-suitesparse --download-spai
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
$PYTHON3_DIR/bin/python3 -m pip install src/binding/petsc4py
