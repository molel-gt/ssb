#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/ADIOS2 ]; then
    echo 'ADIOS2 directory exists'
else
    git clone https://github.com/ornladios/ADIOS2.git
fi

mkdir adios2-build && cd adios2-build


cmake ../ADIOS2 -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DADIOS2_BUILD_EXAMPLES=ON -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON -DADIOS2_USE_PIP=ON -DBUILD_SHARED_LIBS=ON -DADIOS2_USE_Python=ON -DADIOS2_USE_Fortran=ON -DCMAKE_CXX_FLAGS='-std=c++11'

make -j3
make install

echo "export ADIOS2_ROOT=$CMAKE_INSTALL_PREFIX/adios2" >> $HOME/.bashrc
echo "export ADIOS2_DIR=$CMAKE_INSTALL_PREFIX/adios2"  >> $HOME/.bashrc
