#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/mpich-4.3.0b1 ]; then
    echo 'directory exists, skip downloading'
else
    wget https://www.mpich.org/static/downloads/4.3.0b1/mpich-4.3.0b1.tar.gz
    tar xvzf mpich-4.3.0b1.tar.gz
fi

cd $SOFTWARES_DIR/mpich-4.3.0b1
./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
