#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/ufl ]; then
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/FEniCS/ufl.git
fi

cd $SOFTWARES_DIR/ufl
$PYTHON3_DIR/bin/python3 -m pip install $SOFTWARES_DIR/ufl
