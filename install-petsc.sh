#!/bin/bash
cd $HOME/softwares/petsc
./configure --download-f2cblaslapack --with-mpi-dir=~/opt/ --download-hypre --with-64-bit-indices=no --download-metis --download-parmetis --download-ptscotch --download-eigen --download-hdf5 PETSC_ARCH=real-int32 --with-scalar-type=real --with-shared-libraries --with-debugging=no --download-superlu_dist --download-mumps --download-scalapack --download-netcdf --download-zlib --download-ml --download-suitesparse --download-spai --download-strumpack --download-spooles --download-zfp --download-butterflypack --with-log=1
make PETSC_DIR=$HOME/softwares/petsc PETSC_ARCH=real-int32 all
make PETSC_DIR=$HOME/softwares/petsc PETSC_ARCH=real-int32 check
$HOME/python3-env/bin/python3 -m pip install src/binding/petsc4py
