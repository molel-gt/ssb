#!/bin/bash

version=$1
url=https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-$version.tar.gz
OPENMPI_SRC_DIR=$SOFTWARES_DIR/opempi-$version

cd $SOFTWARES_DIR

if [ -d "$OPENMPI_SRC_DIR" ]; then wget url .; fi

tar xvzf $OPENMPI_SRC_DIR.tar.gz
cd $OPENMPI_SRC_DIR
./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j
make install
