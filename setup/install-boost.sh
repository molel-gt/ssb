#!/bin/bash

cd $SOFTWARES_DIR
version=1_86_0
BOOST_DIR=$SOFTWARES_DIR/boost_$version
if [ -f $BOOST_DIR.tar.gz ]; then
    echo "directory exists"
    if [ -d $BOOST_DIR ]; then
        echo "already extracted"
    else
        tar xvzf boost_$version.tar.gz
    fi
else
    wget https://archives.boost.io/release/$(echo $version | tr _ . )/source/boost_$version.tar.gz
    tar xvzf boost_$version.tar.gz
fi

cd $BOOST_DIR
./bootstrap.sh prefix=$CMAKE_INSTALL_PREFIX
./b2 prefix=$CMAKE_INSTALL_PREFIX