#!/bin/bash

echo -e "Enter a seed: "
read seed
echo "Seed is ${seed}"

datacenters="VBFSUSY_13_Higgsino_150_mmjj_500_-1_-${seed} VjjEWK_13_mmjj_1000_4000_-${seed} VjjEWK_13_mmjj_4000_7000_-${seed} VjjEWK_13_mmjj_7000_10000_-${seed} VjjEWK_13_mmjj_10000_-1_-${seed} VjjQCD_13_mmjj_1000_4000_-${seed} VjjQCD_13_mmjj_4000_7000_-${seed} VjjQCD_13_mmjj_7000_10000_-${seed} VjjQCD_13_mmjj_10000_-1_-${seed}"


for files in $datacenters; do
    file=$files
    database=/data/users/jupyter-zdethlof/SUSY/${file}
    cd $database
    echo $PWD
    echo Cross-Section
    grep "Cross-section" docker_mgpy.log | tail -1
    database=/data/users/jupyter-zdethlof/SUSY/${file}/madgraph/PROC_madgraph/Events/run_01
    cd $database
    echo Event Number
    gunzip -c unweighted_events.lhe.gz | grep "<event>" | wc |  awk '{print $1}'
    


done
