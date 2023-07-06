#!/bin/bash

git clone https://github.com/spack/spack.git

. ./spack/share/spack/setup-env.sh

spack env create fenicsx-env
spack env activate fenicsx-env
spack add hypre~int64
spack add metis~int64
spack add ninja@1.10.0
spack add openmpi schedulers=slurm
spack add petsc@3.17.5~int64
spack add py-fenics-dolfinx cflags="-O3" fflags="-O3"
spack add superlu_dist~int64
spack add tetgen

spack install