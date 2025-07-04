#!/bin/bash

DOLFINX_BUILD_DIR=$SOFTWARES_DIR/dolfinx/cpp/build

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/dolfinx ]; then
    cd $SOFTWARES_DIR/dolfinx && git stash && git pull origin
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/FEniCS/dolfinx.git
fi

. $PYTHON3_DIR/bin/activate

cd $SOFTWARES_DIR/dolfinx/cpp/

if [ -d "$DOLFINX_BUILD_DIR" ]; then rm -Rf $DOLFINX_BUILD_DIR; fi
mkdir $DOLFINX_BUILD_DIR

cd $DOLFINX_BUILD_DIR
cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_CXX_FLAGS="-O3"
make && make install
dolfinxconf=$(find $CMAKE_INSTALL_PREFIX/ -name dolfinx.conf -print)
. $dolfinxconf

cd $SOFTWARES_DIR/dolfinx/python

$PYTHON3_DIR/bin/python3 -m pip install -r build-requirements.txt
$PYTHON3_DIR/bin/python3 -m pip install --check-build-dependencies --no-build-isolation .
