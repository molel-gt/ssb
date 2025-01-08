#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/petsc ]; then
    echo 'Directory exists, not cloning'
    git stash && git pull origin
else
    git clone https://gitlab.com/petsc/petsc.git
fi

cd $SOFTWARES_DIR/petsc
git checkout v3.22.2

./configure --with-clean --download-f2cblaslapack --with-mpi-dir=$OPENMPI_DIR --download-hypre --with-64-bit-indices=no --download-metis --download-parmetis --download-ptscotch --download-eigen --download-hdf5 PETSC_ARCH=real-int32 --with-scalar-type=real --with-shared-libraries --with-debugging=no --download-superlu_dist --download-mumps --download-scalapack --with-log=1 --with-cuda=0 --use-gpu-aware-mpi=0  --download-strumpack --download-netcdf --download-zlib --download-ml --download-suitesparse --download-spai --download-spooles --download-zfp --download-butterflypack
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
$PYTHON3_DIR/bin/python3 -m pip install src/binding/petsc4py
