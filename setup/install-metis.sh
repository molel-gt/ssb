#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/METIS ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/METIS.git
fi

cd $SOFTWARES_DIR/METIS
mkdir build
make config shared=1 prefix=$CMAKE_INSTALL_PREFIX gklib_path=$CMAKE_INSTALL_PREFIX
make install