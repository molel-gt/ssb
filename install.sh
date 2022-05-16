#!/usr/bin/env bash
cd /storage/home/hcoda1/0/emolel3/p-tf74-0/
git clone https://github.com/spack/spack.git
. ./spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
spack add gcc@10.3.0
spack install
spack location -i gcc@10.3.0 | xargs spack load
spack compiler add gcc@10.3.0
spack add py-fenics-dolfinx%gcc@10.3.0 cflags="-O3" fflags="-O3"
spack install