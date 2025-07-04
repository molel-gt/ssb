#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/pugixml ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/zeux/pugixml.git
fi

cd $SOFTWARES_DIR/pugixml

cmake . -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j && make install
