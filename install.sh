#!/usr/bin/env bash
cd ./storage/home/hcoda1/0/emolel3/p-tf74-0/
git clone https://github.com/spack/spack.git
. ../spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack add gcc@10.3.0
spack install
spack load gcc@10.3.0
spack location -i gcc@10.3.0 | xargs spack compiler add
spack add py-fenics-dolfinx@main%gcc@10.3.0 cflags="-O3" fflags="-O3"
spack add gmsh@4.8.4%gcc@10.3.0
spack add py-matplotlib
spack add libtiff%gcc@10.3.0
spack install