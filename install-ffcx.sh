#!/bin/bash

cd $SOFTWARES_DIR

FFCX_DIR=$SOFTWARES_DIR/ffcx

. $PYTHON3_DIR/bin/activate

if [ -d "$FFCX_DIR" ]; then
    echo "Skip cloning, directory exists"
else
    git clone https://github.com/FEniCS/ffcx.git .
fi

cd $FFCX_DIR
mkdir $DOLFINX_BUILD_DIR

cd $FFCX_DIR/cpp
mkdir build
cd $FFCX_DIR/cpp/build
cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j && make install

cd $FFCX_DIR/python
$PYTHON3_DIR/bin/python3 -m pip install .

# $PYTHON3_DIR/bin/python3 -m pip install -r build-requirements.txt
# $PYTHON3_DIR/bin/python3 -m pip install --check-build-dependencies --no-build-isolation .
