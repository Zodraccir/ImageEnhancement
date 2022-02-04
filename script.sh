#!/bin/sh

mkdir resultsResNet20k_5e-6_64_0.001
mkdir resultsResNet20k_5e-6_64_0.0005
mkdir resultsResNet20k_5e-6_64_0.00025
mkdir resultsResNet20k_5e-6_64_0.0001

python3 main_ddqn.py -n 20000 -e 5e-6 -b 64 -lr 0.001 > resultsResNet20k_5e-6_64_0.001/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_64_0.001/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_64_0.001/
mv plots/* resultsResNet20k_5e-6_64_0.001/
mv plots_custom/* resultsResNet20k_5e-6_64_0.001/
mv results.csv resultsResNet20k_5e-6_64_0.001/
mv learning_results.csv resultsResNet20k_5e-6_64_0.001/

python3 main_ddqn.py -n 20000 -e 5e-6 -b 64 -lr 0.0005 > resultsResNet20k_5e-6_64_0.0005/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_64_0.0005/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_64_0.0005/
mv plots/* resultsResNet20k_5e-6_64_0.0005/
mv plots_custom/* resultsResNet20k_5e-6_64_0.0005/
mv results.csv resultsResNet20k_5e-6_64_0.0005/
mv learning_results.csv resultsResNet20k_5e-6_64_0.0005/

python3 main_ddqn.py -n 20000 -e 5e-6 -b 64 -lr 0.00025 > resultsResNet20k_5e-6_64_0.00025/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_64_0.00025/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_64_0.00025/
mv plots/* resultsResNet20k_5e-6_64_0.00025/
mv plots_custom/* resultsResNet20k_5e-6_64_0.00025/
mv results.csv resultsResNet20k_5e-6_64_0.00025/
mv learning_results.csv resultsResNet20k_5e-6_64_0.00025/

python3 main_ddqn.py -n 20000 -e 5e-6 -b 64 -lr 0.0001 > resultsResNet20k_5e-6_64_0.0001/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_64_0.0001/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_64_0.0001/
mv plots/* resultsResNet20k_5e-6_64_0.0001/
mv plots_custom/* resultsResNet20k_5e-6_64_0.0001/
mv results.csv resultsResNet20k_5e-6_64_0.0001/
mv learning_results.csv resultsResNet20k_5e-6_64_0.0001/

