#!/bin/bash

STRUMPACK_DIR=$SOFTWARES_DIR/STRUMPACK
STRUMPACK_BUILD_DIR=$STRUMPACK_DIR/build
cd $SOFTWARES_DIR

if [ -d $STRUMPACK_DIR ]; then
    echo "directory exists"
else:
    git clone https://github.com/pghysels/STRUMPACK.git
fi

cd $STRUMPACK_DIR
if [ -d $STRUMPACK_BUILD_DIR ]; then
    rm -r build
fi

mkdir build
cd $STRUMPACK_BUILD_DIR
METIS_DIR=$CMAKE_INSTALL_PREFIX cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j4
make install