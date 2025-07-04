#!/bin/bash
version=5.0.8
OPENMPI_SRC_DIR=$SOFTWARES_DIR/openmpi-$version

cd $SOFTWARES_DIR

if [ -d "$OPENMPI_SRC_DIR" ]; then
    echo 'directory exists, skip cloning'
else
    if [ -d "$OPENMPI_SRC_DIR".tar.gz ]; then
        echo "zipped file exists"
    else
        wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-$version.tar.gz
    fi
    tar xvzf openmpi-$version.tar.gz
fi

cd $OPENMPI_SRC_DIR
./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
