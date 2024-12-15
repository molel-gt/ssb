#!/bin/bash

rpm -qa
if [ $? -eq 0 ]; then
    echo "HPC environment, loading modules"
    sh modules.sh
else
    echo "NOT HPC environment"
fi

sh install-pkgconf.sh
sh install-pugixml.sh
sh install-adios2.sh
sh install-petsc.sh
sh install-slepc.sh
sh install-basix.sh
sh install-ffcx.sh

$PYTHON3_DIR/bin/python3 -m pip install ufl
