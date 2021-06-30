#!/bin/bash

hubS=/data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Signal/150mjj/*
hubBE=/data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Background/EWKBackground/*
hubBQ=/data/users/jupyter-blonsbro/SUSY/Generations/13TeV/Background/QCDBackground/*

for file in $hubS; do
    mv ${file}/analysis/SimpleAna.root ${file}/analysis/histograms.root
    echo $file
    ls ${file}/analysis

done

for file in $hubBE; do
    mv ${file}/analysis/SimpleAna.root ${file}/analysis/histograms.root
    echo $file
    ls ${file}/analysis

done

for file in $hubBQ; do
    mv ${file}/analysis/SimpleAna.root ${file}/analysis/histograms.root
    echo $file
    ls ${file}/analysis

done
