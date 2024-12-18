#!/bin/bash

OPENMPI_SRC_DIR=$SOFTWARES_DIR/openmpi-5.0.5

cd $SOFTWARES_DIR

if [ -d "$OPENMPI_SRC_DIR" ]; then
    echo 'directory exists, skip cloning'
else
    # git clone https://github.com/open-mpi/ompi.git --recursive
    wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
    tar xvzf openmpi-5.0.5.tar.gz
fi

# git checkout v5.0.5

cd $OPENMPI_SRC_DIR
# ./autogen.pl
./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
