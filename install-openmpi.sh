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
./configure --prefix=$CMAKE_INSTALL_PREFIX --enable-shared --with-pmix=internal --with-hwloc=embedded --without-cuda --without-ze --with-cuda=no --with-ze=no --enable-mca-no-build=pgpu --enable-mpi-ext=affinity,ftmpi,rocm,shortfloat --enable-mca-dso=null,rocm,ze
make -j3
make install
