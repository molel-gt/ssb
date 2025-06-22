#!/bin/bash

cd $SOFTWARES_DIR

if [ -d $SOFTWARES_DIR/ufl ]; then
    # cd $SOFTWARES_DIR/ufl && git stash && git pull origin
    echo 'directory exists, skip cloning'
else
    git clone https://github.com/FEniCS/ufl.git
fi

cd $SOFTWARES_DIR/ufl
$PYTHON3_DIR/bin/python3 -m pip install $SOFTWARES_DIR/ufl
