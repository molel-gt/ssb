#!/bin/bash

SOFTWARES_DIR=$HOME/softwares
DOLFINX_BUILD_DIR=$HOME/softwares/dolfinx/cpp/build

. $HOME/python3-env/bin/activate

cd $HOME/softwares/dolfinx/cpp/

if [ -d "$DOLFINX_BUILD_DIR" ]; then rm -Rf $DOLFINX_BUILD_DIR; fi
mkdir $DOLFINX_BUILD_DIR

cd $DOLFINX_BUILD_DIR
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/opt -DCMAKE_PREFIX_PATH=$HOME/opt
make && make install
. $HOME/opt/lib/dolfinx/dolfinx.conf

cd $HOME/softwares/dolfinx/python

$HOME/python3-env/bin/python3 -m pip install -r build-requirements.txt
$HOME/python3-env/bin/python3 -m pip install --check-build-dependencies --no-build-isolation .
