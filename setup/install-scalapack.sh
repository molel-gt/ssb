#!/bin/bash

scalapack_DIR=$SOFTWARES_DIR/scalapack
scalapack_BUILD_DIR=$scalapack_DIR/build
cd $SOFTWARES_DIR

if [ -d $scalapack_DIR ]; then
    echo "directory exists"
else:
    git clone https://github.com/Reference-ScaLAPACK/scalapack/
fi

cd $scalapack_DIR
if [ -d $scalapack_BUILD_DIR ]; then
    rm -r build
fi

mkdir build
cd $scalapack_BUILD_DIR
METIS_DIR=$CMAKE_INSTALL_PREFIX cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j4
make install