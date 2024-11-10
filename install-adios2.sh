#!/bin/bash

cd $HOME/softwares

if -d $HOME/softwares/ADIOS2; then
    echo 'ADIOS2 directory exists'
else
    git clone https://github.com/ornladios/ADIOS2.git ADIOS2
fi

mkdir adios2-build && cd adios2-build


cmake ../ADIOS2 -DCMAKE_PREFIX_PATH=$HOME/opt -DCMAKE_INSTALL_PREFIX=$HOME/opt -DADIOS2_BUILD_EXAMPLES=ON -DADIOS2_USE_MPI=ON -DADIOS2_USE_HDF5=ON -DBUILD_SHARED_LIBS=ON -DADIOS2_USE_Python=ON -DADIOS2_USE_Fortran=ON -DCMAKE_CXX_FLAGS='-std=c++11'

make -j3
make install
