#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/slepc ]; then
    echo "directory exists, skip cloning"
else
    git clone https://gitlab.com/slepc/slepc
fi
cd $SOFTWARES_DIR/slepc
git stash && git checkout v3.22.2
./configure --prefix=$CMAKE_INSTALL_PREFIX --with-clean
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH
make SLEPC_DIR=$SLEPC_DIR PETSC_DIR=$PETSC_DIR install
$PYTHON3_DIR/bin/python3 -m pip install src/binding/slepc4py
