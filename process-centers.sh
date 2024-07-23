#!/usr/bin/env bash

for eps in 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
    echo 'Generating centers'
    cp centers/$eps.txt .
    ./spheres $eps.txt
    cp $eps.csv centers
    echo 'Processing file 'centers/$eps.csv
    sed -i '1,6d' centers/$eps.csv
    sed -i 's/[ \t]*$//' centers/$eps.csv
    sed -i 's/\ /,/g' centers/$eps.csv
    sed -i '1i x,y' centers/$eps.csv
    rm $eps.txt
done
