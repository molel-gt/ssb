#!/bin/bash

$PYTHON3_DIR/bin/python3 -m pip uninstall scikit-build-core nanobind
sh install-ufl
sh install-ffcx
sh install-basix
sh install-fenicsx
