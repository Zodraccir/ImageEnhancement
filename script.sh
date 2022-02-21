#!/bin/sh

mkdir resultsAlex20k_2e-5_64_0.0001
mkdir resultsAlex20k_2e-5_64_0.001
mkdir resultsAlex20k_2e-5_64_0.0005
mkdir resultsAlex20k_2e-5_256_0.001
mkdir resultsAlex20k_2e-5_256_0.0001
mkdir resultsAlex20k_2e-5_256_0.0005

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0001> resultsAlex20k_2e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_64_0.0001/
mv plots/* resultsAlex20k_2e-5_64_0.0001/
mv plots_custom/* resultsAlex20k_2e-5_64_0.0001/
mv results.csv resultsAlex20k_2e-5_64_0.0001/
mv learning_results.csv resultsAlex20k_2e-5_64_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.001> resultsAlex20k_2e-5_64_0.001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.001/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_64_0.001/
mv plots/* resultsAlex20k_2e-5_64_0.001/
mv plots_custom/* resultsAlex20k_2e-5_64_0.001/
mv results.csv resultsAlex20k_2e-5_64_0.001/
mv learning_results.csv resultsAlex20k_2e-5_64_0.001/


python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0005> resultsAlex20k_2e-5_64_0.0005/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.0005/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_64_0.0005/
mv plots/* resultsAlex20k_2e-5_64_0.0005/
mv plots_custom/* resultsAlex20k_2e-5_64_0.0005/
mv results.csv resultsAlex20k_2e-5_64_0.0005/
mv learning_results.csv resultsAlex20k_2e-5_64_0.0005/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 256 -lr 0.0001> resultsAlex20k_2e-5_256_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_256_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_256_0.0001/
mv plots/* resultsAlex20k_2e-5_256_0.0001/
mv plots_custom/* resultsAlex20k_2e-5_256_0.0001/
mv results.csv resultsAlex20k_2e-5_256_0.0001/
mv learning_results.csv resultsAlex20k_2e-5_256_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 256 -lr 0.001> resultsAlex20k_2e-5_256_0.001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_256_0.001/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_256_0.001/
mv plots/* resultsAlex20k_2e-5_256_0.001/
mv plots_custom/* resultsAlex20k_2e-5_256_0.001/
mv results.csv resultsAlex20k_2e-5_256_0.001/
mv learning_results.csv resultsAlex20k_2e-5_256_0.001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 256 -lr 0.0005> resultsAlex20k_2e-5_256_0.0005/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-5_256_0.0005/logTest40k.dat
mv models/image* resultsAlex20k_2e-5_256_0.0005/
mv plots/* resultsAlex20k_2e-5_256_0.0005/
mv plots_custom/* resultsAlex20k_2e-5_256_0.0005/
mv results.csv resultsAlex20k_2e-5_256_0.0005/
mv learning_results.csv resultsAlex20k_2e-5_256_0.0005/


