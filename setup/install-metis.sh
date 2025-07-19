#!/bin/bash
version=v5.21
cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/METIS ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/METIS.git
fi

cd $SOFTWARES_DIR/METIS
git checkout $version
# rm -rf build-dir
# mkdir build-dir
# cd build-dir
# cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_POLICY_VERSION_MINIMUM=3.28
make config shared=1 prefix=$CMAKE_INSTALL_PREFIX gklib_path=$CMAKE_INSTALL_PREFIX
make install
