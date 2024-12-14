#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/spdlog ]; then;
    echo 'directory exists, skip cloning'
else;
    git clone https://github.com/gabime/spdlog.git
fi

cd $SOFTWARES_DIR/spdlog
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j && make install
