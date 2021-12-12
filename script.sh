#!/bin/sh
python3 --version
python3 main_ddqn.py -n 4000 -e 1e-4 > logTraining4k.dat
python3 bestPass.py > logTest4k.dat
rm models/*
python3 main_ddqn.py -n 8000 -e 6e-5 > logTraining8k.dat
python3 bestPass.py > logTest8k.dat
rm models/*
python3 main_ddqn.py -n 12000 -e 3e-5 > logTraining12k.dat
python3 bestPass.py > logTest12k.dat
rm models/*
python3 main_ddqn.py -n 16000 -e 2e-5 > logTraining16k.dat
python3 bestPass.py > logTest16k.dat
rm models/*
python3 main_ddqn.py -n 20000 -e 1e-5 > logTraining20k.dat
python3 bestPass.py > logTest20k.dat
rm models/*
python3 main_ddqn.py -n 40000 -e 6e-6 > logTraining40k.dat
python3 bestPass.py > logTest40k.dat