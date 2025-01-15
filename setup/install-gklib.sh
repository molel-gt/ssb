#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/GKlib ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/GKlib.git
fi

cd $SOFTWARES_DIR/GKlib
make config cc=mpicc prefix=$CMAKE_INSTALL_PREFIX
make install