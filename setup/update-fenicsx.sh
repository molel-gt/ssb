#!/bin/bash

$PYTHON3_DIR/bin/python3 -m pip uninstall scikit-build-core -y nanobind -y
cd $WORK_DIR/setup
sh install-ufl.sh
sh install-ffcx.sh
sh install-basix.sh
sh install-fenicsx.sh
