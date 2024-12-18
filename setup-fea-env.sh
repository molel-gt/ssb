#!/bin/bash

sh install-autoconf.sh
sh install-openmpi.sh
sh install-pkgconf.sh
sh install-pugixml.sh
sh install-spdlog.sh
sh install-adios2.sh
sh install-petsc.sh
sh install-slepc.sh
sh install-basix.sh
sh install-ufl.sh
sh install-ffcx.sh
sh install-hdf5.sh
sh install-fenicsx.sh

$PYTHON3_DIR/bin/python3 -m pip install gmsh scipy matplotlib
