#!/bin/bash

cd $SOFTWARES_DIR
version=13.3.0
if [ -d $SOFTWARES_DIR/gcc-$version ]; then
    echo 'Folder exists'
else
    wget https://mirrors.ocf.berkeley.edu/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.gz #https://ftp.gnu.org/gnu/gcc/gcc-$version/gcc-$version.tar.gz
    tar xvzf gcc-$version.tar.gz
fi

cd gcc-$version

./configure --prefix=$CMAKE_INSTALL_PREFIX --enable-languages=c,c++,fortran,go --disable-multilib
make
make install
