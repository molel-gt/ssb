#!/usr/bin/env bash
echo "Installing dolfinx.."
# boost
# cd $HOME
# wget https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz
# hash=${sha256sum boost_1_78_0.tar.gz} 
# if hash != 94ced8b72956591c4775ae2207a9763d3600b30d9d7446562c552f0a14a63be7; then
#     echo "file corrupted"
#     exit 1
# fi
# tar xvzf boost_1_78_0.tar.gz
# cd boost_1_78_0

# eigen3
cd $HOME
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xvzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ ../
make install

# pkg-config
cd $HOME
wget https://pkgconfig.freedesktop.org/releases/pkg-config-0.29.2.tar.gz
tar xvzf pkg-config-0.29.2.tar.gz
cd pkg-config-0.29.2
./configure --prefix=/opt/
make && make install

# hdf5
cd $HOME
wget https://www.hdfgroup.org/package/hdf5-1-12-1-tar-gz/?wpdmdl=15727&refresh=62439b2b5d5f51648597803
hsf5hash=`sha256sum hdf5-1-12-1.tar.gz`
if $hdf5hash != 79c66ff67e666665369396e9c90b32e238e501f345afd2234186bfb8331081ca; then
    echo "file is corrupted"
    exit 1
fi
tar xvzf hdf5-1-12-1.tar.gz
cd hdf5-1-12-1

# petsc
cd $HOME
# git clone -b release https://gitlab.com/petsc/petsc.git petsc
wget https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/petsc-3.16.5.tar.gz
tar xvzf petsc-3.16.5.tar.gz
cd petsc
./configure --prefix=/opt/  --download-parmetis --download-ptscotch --download-suitesparse --download-mumps --download-hypre

# parmetis
cd $HOME

# scotch
cd $HOME

# slepc
cd $HOME

# ffc-x
cd $HOME
git clone https://github.com/FEniCS/ffcx.git
cd ffcx
$ cmake -B build-dir -S cmake/
$ cmake --build build-dir
$ cmake --install build-dir
cd $HOME/ffcx
python3  -m pip install . --user

# ufl
cd $HOME
https://github.com/FEniCS/ufl.git
cd ufl
python3 -m pip install . --user

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

# basix
cd $HOME
git clone https://github.com/FEniCS/basix.git
cd basix/cpp
cmake -DCMAKE_BUILD_TYPE=Release -B build-dir -S .
cmake --build build-dir
cmake --install build-dir
cd ../python
python3 -m pip install . --user

# pip installs
cd $HOME
python3 -m pip install pybind11 numpy mpi4py petsc4py matplotlib slepc4py --user


# install dolfinx
cd $HOME
cd dolfinx/cpp/
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ ../
make install

cd ../../python && python3 -m pip install . --user

echo "successfully installed dolfinx"