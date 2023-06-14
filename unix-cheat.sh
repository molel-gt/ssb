#!/bin/bash

# list files matching pattern and display results
find . -name \*.msh -print
find . -name \*unscaled* -print
# must install tree via sudo apt install tree
tree -if --noreport .
# split character
squeue -u emolel3 | awk '{split($0,a," "); print a[1]}' | xargs scancel

# list file in human-readable
ls -lh