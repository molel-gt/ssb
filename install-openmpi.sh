#!/bin/bash

OPENMPI_SRC_DIR=$SOFTWARES_DIR/ompi

cd $SOFTWARES_DIR

if [ -d "$OPENMPI_SRC_DIR" ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/open-mpi/ompi.git --recursive
fi

git checkout v5.0.6

cd $OPENMPI_SRC_DIR
./autogen.pl
./configure --prefix=$CMAKE_INSTALL_PREFIX --disable-cuda
make -j3
make install
