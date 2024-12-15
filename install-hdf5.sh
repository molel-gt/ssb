#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/hdf5 ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/HDFGroup/hdf5.git
fi

cd $SOFTWARES_DIR/hdf5
./autogen.sh
CC=$(which mpicc) ./configure --enable-parallel --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
