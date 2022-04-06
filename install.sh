#!/usr/bin/env bash
set -e
echo "Installing dolfinx.."

# eigen3
cd $HOME
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ ..
make install

# pkg-config
cd $HOME
wget https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar xvzf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --prefix=/opt/ --with-internal-glib
make && make install

# petsc
cd $HOME
git clone -b release https://gitlab.com/petsc/petsc.git
cd petsc
./configure --prefix=/opt/  PETSC_ARCH=linux-gnu --download-metis --download-parmetis --download-ptscotch --download-suitesparse --download-scalapack --download-mumps --download-hypre --download-fblaslapack
make PETSC_DIR=$HOME/petsc PETSC_ARCH=linux-gnu && make install
export PETSC_DIR=/opt

# slepc
cd $HOME
wget https://slepc.upv.es/download/distrib/slepc-3.16.2.tar.gz
slepchash=`md5sum slepc-3.16.2.tar.gz | awk '{split($0,a," "); print a[1]}'`
if $slepchash != 673dbda220e5a4bd2c3a6618267d8e55; then
    echo "file corrupted"
    exit 1
fi
tar xvzf slepc-3.16.2.tar.gz
cd slepc-3.16.2
./configure --prefix=/opt/
make && make install

# ufl
cd $HOME
git clone https://github.com/FEniCS/ufl.git
cd ufl
python3 -m pip install . --user

# basix
cd $HOME
git clone https://github.com/FEniCS/basix.git
cd basix/cpp
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
cd $HOME/basix
python3 -m pip install . --user

# ffc-x
cd $HOME
git clone https://github.com/FEniCS/ffcx.git
cd ffcx
cmake -B build-dir -S cmake/
cmake --build build-dir
cmake --install build-dir
cd $HOME/ffcx
python3  -m pip install . --user

# ADIOS2
cd $HOME
git clone https://github.com/ornladios/ADIOS2.git
cd ADIOS2
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ ..
make -j 16
make install

# xtl
cd $HOME
git clone https://github.com/xtensor-stack/xtl.git
cd xtl
cmake -DCMAKE_INSTALL_PREFIX=/opt/
make install

# xtensor
cd $HOME
git clone https://github.com/xtensor-stack/xtensor.git
cd xtensor
cmake -DCMAKE_INSTALL_PREFIX=/opt/
make install



# pip installs
cd $HOME
python3 -m pip install pybind11 numpy mpi4py petsc4py matplotlib slepc4py --user


# install dolfinx
cd $HOME
git clone https://github.com/FEniCS/dolfinx.git
cd dolfinx/cpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ ..
make install

cd ../../python && python3 -m pip install . --user

echo "successfully installed dolfinx"
