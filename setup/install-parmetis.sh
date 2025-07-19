#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/ParMETIS ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/ParMETIS.git
fi

cd $SOFTWARES_DIR/ParMETIS
# git pull origin
# rm -rf build
# mkdir build
# cd build
# cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_POLICY_VERSION_MINIMUM=3.28 -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make config shared=1 prefix=$CMAKE_INSTALL_PREFIX gklib_path=$CMAKE_INSTALL_PREFIX
make install
