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
sh install-ufl.sh
sh install-ffcx.sh
export CC=$CMAKE_INSTALL_PREFIX/bin/mpicc
sh install-hdf5.sh
