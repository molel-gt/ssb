#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/spdlog ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/gabime/spdlog.git
fi

git checkout master && git checkout v1.11.0

cd $SOFTWARES_DIR/spdlog
if [ -d $SOFTWARES_DIR/spdlog/build ]; then
    rm -rf $SOFTWARES_DIR/spdlog/build
fi
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH -DCMAKE_CXX_FLAGS='-fPIC' -DSPDLOG_USE_STD_FORMAT=ON
make -j && make install
