#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/pkg-config ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://gitlab.freedesktop.org/pkg-config/pkg-config.git
fi

cd $SOFTWARES_DIR/pkg-config
git checkout pkg-config-0.29.2
./autogen.sh --prefix=$CMAKE_INSTALL_PREFIX --with-internal-glib
./configure --prefix=$CMAKE_INSTALL_PREFIX
cmake . -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j && make install
