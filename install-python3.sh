#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/Python-3.12.3 ]; then
    echo 'directory exists, not downloading'
else
    wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz
fi
tar xvf Python-3.12.3.tgz

./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
