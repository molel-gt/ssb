#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/ADIOS2 ]; then
    echo 'ADIOS2 directory exists'
    rm -rf adios2-build
else
    git clone https://github.com/ornladios/ADIOS2.git
fi
cd $SOFTWARES_DIR/ADIOS2
git checkout v2.10.2
cd $SOFTWARES_DIR

mkdir -p adios2-build && cd adios2-build

. ~/python3-env/bin/activate
cmake ../ADIOS2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DADIOS2_BUILD_EXAMPLES=ON -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON -DBUILD_SHARED_LIBS=ON -DADIOS2_USE_Python=ON -DADIOS2_USE_Fortran=ON -DCMAKE_CXX_FLAGS='-std=c++11'

make -j3
make install
