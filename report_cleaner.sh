#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "requires file path as parameter"
    exit 1
fi

sed -i '/fenicsx/d' $1
sed -i '/ffcx/d' $1
sed -i '/build_ext/d' $1
sed -i '/root/d' $1