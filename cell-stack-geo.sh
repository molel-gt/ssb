#!/bin/bash

echo 'Generating mesh for lithium plating in cell stack'
python3 cell_stack_geo.py && gmsh mesh.msh
