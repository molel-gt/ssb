#!/bin/bash

sh install-pkgconf.sh
sh install-pugixml.sh
sh install-adios2.sh
sh install-petsc.sh
sh install-slepc.sh
sh install-basix.sh
sh install-ufl.sh
sh install-ffcx.sh
# sh install-autoconf.sh
sh install-hdf5.sh
$PYTHON3_DIR/bin/python3 -m pip install python-gmsh
sh install-fenicsx.sh
