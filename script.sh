#!/bin/sh
python3 --version

python3 main_ddqn.py -n 20000 -e 1e-5 > logTraining20k.dat
python3 bestPass.py > logTest20k.dat
rm models/*
python3 main_ddqn.py -n 40000 -e 9e-6 > logTraining40k.dat
python3 bestPass.py > logTest40k.dat
rm models/*
python3 main_ddqn.py -n 80000 -e 5e-6 > logTraining80k.dat
python3 bestPass.py > logTest80k.dat
#rm models/*
#python3 main_ddqn.py -n 160000 -e 5e-6 > logTraining160k.dat
#python3 bestPass.py > logTest160k.dat