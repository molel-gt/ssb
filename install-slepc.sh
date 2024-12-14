#!/bin/bash
cd $SOFTWARES_DIR
./configure --prefix=$CMAKE_INSTALL_PREFIX --with-clean
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR install
$PYTHON3_DIR/bin/python3 -m pip install src/binding/slepc4py
