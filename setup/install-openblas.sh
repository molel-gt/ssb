#!/bin/bash

OpenBLAS_DIR=$SOFTWARES_DIR/OpenBLAS
OpenBLAS_BUILD_DIR=$OpenBLAS_DIR/build
cd $SOFTWARES_DIR

if [ -d $OpenBLAS_DIR ]; then
    echo "directory exists"
else:
    git clone https://github.com/OpenMathLib/OpenBLAS.git
fi

cd $OpenBLAS_DIR
if [ -d $OpenBLAS_BUILD_DIR ]; then
    rm -r build
fi

mkdir build
cd $OpenBLAS_BUILD_DIR
METIS_DIR=$CMAKE_INSTALL_PREFIX cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make PREFIX=$CMAKE_INSTALL_PREFIX
make PREFIX=$CMAKE_INSTALL_PREFIX install