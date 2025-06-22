#!/bin/bash

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/ParMETIS ]; then
    echo 'directory exists, not cloning'
else
    git clone https://github.com/KarypisLab/ParMETIS.git
fi

cd $SOFTWARES_DIR/ParMETIS
make config shared=1 prefix=$CMAKE_INSTALL_PREFIX
make install