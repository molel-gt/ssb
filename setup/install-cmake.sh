#/bin/bash

version=4.1.1

cd $SOFTWARES_DIR
if [ -d $SOFTWARES_DIR/cmake-$version ]; then
    echo 'Folder exists'
else:
    wget https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version.tar.gz .
    tar xvzf cmake-$version.tar.gz
fi

cd cmake-$version

.configure --prefix=$CMAKE_INSTALL_PREFIX

gmake -j
gmake install
