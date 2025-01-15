#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/pkgconf ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/pkgconf/pkgconf.git
fi

cd $SOFTWARES_DIR/pkgconf
./autogen.sh
./configure --prefix=$CMAKE_INSTALL_PREFIX
# cmake . -DCMAKE_INSTALL_PREFIX=$CMAKE_INSTALL_PREFIX -DCMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH
make -j && make install

echo "export PKG_CONFIG=$CMAKE_INSTALL_PREFIX/bin/pkgconf" >> $HOME/.bashrc
