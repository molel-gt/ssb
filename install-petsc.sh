#!/bin/bash
cd $SOFTWARES_DIR
./configure --download-f2cblaslapack --with-mpi-dir=~/opt/ --download-hypre --with-64-bit-indices=no --download-metis --download-parmetis --download-ptscotch --download-eigen --download-hdf5 PETSC_ARCH=real-int32 --with-scalar-type=real --with-shared-libraries --with-debugging=no --download-superlu_dist --download-mumps --download-scalapack --download-netcdf --download-zlib --download-ml --download-suitesparse --download-spai --download-strumpack --download-spooles --download-zfp --download-butterflypack --with-log=1
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
$HOME/python3-env/bin/python3 -m pip install src/binding/petsc4py
