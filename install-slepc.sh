#!/bin/bash
cd $HOME/softwares/slepc
./configure --prefix=$HOME/opt
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
