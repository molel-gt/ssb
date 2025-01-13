#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/scotch ]; then
    echo 'directory exists, not cloning'
else
    git clone https://gitlab.inria.fr/scotch/scotch.git
fi

cd $SOFTWARES_DIR/scotch

mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DMPI_HOME=$SOFTWARES_DIR/openmpi-5.0.5
make -j5 DESTDIR=$CMAKE_INSTALL_PREFIX && make install DESTDIR=$CMAKE_INSTALL_PREFIX 