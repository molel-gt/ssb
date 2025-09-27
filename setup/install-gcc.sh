#!/bin/bash

cd $SOFTWARES_DIR
version=13.3.0
if [ -d $SOFTWARES_DIR/gcc-$version ]; then
    echo 'Folder exists'
else
    wget https://ftp.gnu.org/gnu/gcc/gcc-$version/gcc-$version.tar.gz
    tar xvzf gcc-$version.tar.gz
fi

cd gcc-$version
