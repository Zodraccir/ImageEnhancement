#!/bin/sh

mkdir resultsAlex20k_1e-5_64_0.002
mkdir resultsAlex20k_5e-5_64_0.002
mkdir resultsAlex20k_2e-5_64_0.002
mkdir resultsAlex20k_1e-4_64_0.002

python3 main_ddqn.py -n 20000 -e 1e-5 -b 64 -lr 0.002> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-5_64_0.002/logTest20k.dat
mv models/image* resultsAlex20k_1e-5_64_0.002/
mv plots/* resultsAlex20k_1e-5_64_0.002/
mv plots_custom/* resultsAlex20k_1e-5_64_0.002/
mv results.csv resultsAlex20k_1e-5_64_0.002/


python3 main_ddqn.py -n 20000 -e 5e-5 -b 64 -lr 0.002> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_5e-5_64_0.002/logTest20k.dat
mv models/image* resultsAlex20k_5e-5_64_0.002/
mv plots/* resultsAlex20k_5e-5_64_0.002/
mv plots_custom/* resultsAlex20k_5e-5_64_0.002/
mv results.csv resultsAlex20k_5e-5_64_0.002/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.002> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.002/logTest20k.dat
mv models/image* resultsAlex20k_2e-5_64_0.002/
mv plots/* resultsAlex20k_2e-5_64_0.002/
mv plots_custom/* resultsAlex20k_2e-5_64_0.002/
mv results.csv resultsAlex20k_2e-5_64_0.002/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.002> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-4_64_0.002/logTest20k.dat
mv models/image* resultsAlex20k_1e-4_64_0.002/
mv plots/* resultsAlex20k_1e-4_64_0.002/
mv plots_custom/* resultsAlex20k_1e-4_64_0.002/
mv results.csv resultsAlex20k_1e-4_64_0.002/