#!/bin/bash

echo -e "Enter seed: "
read seed
echo "Seed: ${seed}"

signal="VBFSUSY_13_Higgsino_150_mmjj_500_-1_-${seed}"
cd /data/users/jupyter-zdethlof/SUSY
echo ${signal}
mv ${signal} /data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Signal/150mjj

datacentersEWK="VjjEWK_13_mmjj_1000_4000_-${seed} VjjEWK_13_mmjj_4000_7000_-${seed} VjjEWK_13_mmjj_7000_10000_-${seed} VjjEWK_13_mmjj_10000_-1_-${seed}"

for files in $datacentersEWK; do
    file=$files
    cd /data/users/jupyter-zdethlof/SUSY
    echo ${file}
    mv ${file} /data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Background/EWKBackground

done

datacentersQCD="VjjQCD_13_mmjj_1000_4000_-${seed} VjjQCD_13_mmjj_4000_7000_-${seed} VjjQCD_13_mmjj_7000_10000_-${seed} VjjQCD_13_mmjj_10000_-1_-${seed}"

for files in $datacentersQCD; do
    file=$files
    cd /data/users/jupyter-zdethlof/SUSY
    echo ${file}
    mv ${file} /data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Background/QCDBackground
done
