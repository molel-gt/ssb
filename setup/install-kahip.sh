#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/KaHIP ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/KaHIP/KaHIP.git
fi

cd $SOFTWARES_DIR/KaHIP
KAHIP_BUILD_DIR=$SOFTWARES_DIR/KaHIP/build

if [ -d "$KAHIP_BUILD_DIR" ]; then rm -Rf $KAHIP_BUILD_DIR; fi
mkdir $KAHIP_BUILD_DIR && cd $KAHIP_BUILD_DIR

cmake ../ -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=$CMAKE_INSTALL_PREFIX/include
make -j5 && make install