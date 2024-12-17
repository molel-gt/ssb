#!/bin/bash

DOLFINX_BUILD_DIR=$SOFTWARES_DIR/dolfinx/cpp/build

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/dolfinx ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/FEniCS/dolfinx.git
fi

. $PYTHON3_DIR/bin/activate

cd $SOFTWARES_DIR/dolfinx/cpp/

if [ -d "$DOLFINX_BUILD_DIR" ]; then rm -Rf $DOLFINX_BUILD_DIR; fi
mkdir $DOLFINX_BUILD_DIR

cd $DOLFINX_BUILD_DIR
CC=$CC CPP=$CPP CXX=$CXX FC=$FC cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make && make install
dolfinxconf=$(find $CMAKE_INSTALL_PREFIX/ -name dolfinx.conf -print)
. $dolfinxconf

cd $SOFTWARES_DIR/dolfinx/python

CC=$CC CPP=$CPP CXX=$CXX FC=$FC $PYTHON3_DIR/bin/python3 -m pip install -r build-requirements.txt
CC=$CC CPP=$CPP CXX=$CXX FC=$FC $PYTHON3_DIR/bin/python3 -m pip install --check-build-dependencies --no-build-isolation .
