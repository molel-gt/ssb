#!/usr/bin/env bash
cd /storage/home/hcoda1/0/emolel3/p-tf74-0/
git clone https://github.com/spack/spack.git
cp /storage/home/hcoda1/0/emolel3/p-tf74-0/config.yaml /storage/home/hcoda1/0/emolel3/p-tf74-0/spack/etc/spack
. /storage/home/hcoda1/0/emolel3/p-tf74-0/spack/share/spack/setup-env.sh
spack env create fenicsx-env
spack env activate fenicsx-env
module load gcc/10.1.0
spack compiler find
spack add py-fenics-dolfinx%gcc@10.1.0 cflags="-O3" fflags="-O3"
spack install