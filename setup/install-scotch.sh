#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/scotch ]; then
    echo 'directory exists, not cloning'
    git pull origin
else
    git clone https://gitlab.inria.fr/scotch/scotch.git
fi

cd $SOFTWARES_DIR/scotch
git checkout v7.0.7

if [ -d build ]; then
    rm -r build
fi
mkdir build && cd build
cmake ..  -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DMPI_HOME=$SOFTWARES_DIR/openmpi-5.0.5
make -j5 && make install
