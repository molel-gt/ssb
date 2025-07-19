#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/GKlib ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/GKlib.git
fi

cd $SOFTWARES_DIR/GKlib
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.28
make config prefix=$CMAKE_INSTALL_PREFIX openmp=set shared=1
make install
