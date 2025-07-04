#!/bin/bash

sh install-autoconf.sh
sh install-openmpi.sh
# sh install-pkgconf.sh
# sh install-pugixml.sh
sh install-spdlog.sh
sh install-hdf5.sh
sh install-adios2.sh
sh install-gklib.sh
sh install-metis.sh
sh install-parmetis.sh
sh install-scotch.sh
sh install-kahip.sh
sh install-openblas.sh
sh install-scalapack.sh
sh install-strumpack.sh
sh install-petsc.sh
sh install-slepc.sh
sh install-basix.sh
sh install-ufl.sh
sh install-ffcx.sh
sh install-fenicsx.sh

$PYTHON3_DIR/bin/python3 -m pip install gmsh scipy matplotlib
