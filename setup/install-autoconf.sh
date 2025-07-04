#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/autoconf ]; then
    echo 'directory exists, skip cloning'
else
    git clone git://git.sv.gnu.org/autoconf
fi
git stash && git checkout master && git pull origin && git checkout v2.72
cd $SOFTWARES_DIR/autoconf
./bootstrap
./configure --prefix=$CMAKE_INSTALL_PREFIX
make -j && make install
