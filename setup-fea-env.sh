#!/bin/bash

rpm -qa
if [ $? -eq 0 ]; then
    echo "HPC environment, loading modules"
    sh modules.sh
else
    echo "NOT HPC environment"
fi
