#!/bin/bash
cd $HOME/softwares/slepc
./configure --prefix=$HOME/opt --with-clean
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR install
$HOME/python3-env/bin/python3 -m pip install src/binding/slepc4py
