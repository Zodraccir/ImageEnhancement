#!/bin/sh

mkdir resultsResnet20k_8e-5_64_0.0001
mkdir resultsResnet20k_1e-4_64_0.0001
mkdir resultsResnet20k_2e-4_64_0.0001
mkdir resultsResnet20k_4e-4_64_0.0001
mkdir resultsResnet10k_2e-4_64_0.0001
mkdir resultsResnet10k_5e-4_64_0.0001
mkdir resultsResnet10k_7e-4_64_0.0001
mkdir resultsResnet10k_9e-4_64_0.0001
mkdir resultsResnet40k_4e-5_64_0.0001
mkdir resultsResnet40k_1e-5_64_0.0001
mkdir resultsResnet40k_2e-5_64_0.0001
mkdir resultsResnet40k_8e-6_64_0.0001


python3 main_ddqn.py -n 20000 -e 8e-5 -b 64 -lr 0.0001> resultsResnet20k_8e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet20k_8e-5_64_0.0001/logTest40k.dat
mv models/image* resultsResnet20k_8e-5_64_0.0001/
mv plots/* resultsResnet20k_8e-5_64_0.0001/
mv plots_custom/* resultsResnet20k_8e-5_64_0.0001/
mv learning_results.csv resultsResnet20k_8e-5_64_0.0001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0001> resultsResnet20k_1e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet20k_1e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet20k_1e-4_64_0.0001/
mv plots/* resultsResnet20k_1e-4_64_0.0001/
mv plots_custom/* resultsResnet20k_1e-4_64_0.0001/
mv learning_results.csv resultsResnet20k_1e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-4 -b 64 -lr 0.0001> resultsResnet20k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet20k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet20k_2e-4_64_0.0001/
mv plots/* resultsResnet20k_2e-4_64_0.0001/
mv plots_custom/* resultsResnet20k_2e-4_64_0.0001/
mv learning_results.csv resultsResnet20k_2e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 4e-4 -b 64 -lr 0.0001> resultsResnet20k_4e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet20k_4e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet20k_4e-4_64_0.0001/
mv plots/* resultsResnet20k_4e-4_64_0.0001/
mv plots_custom/* resultsResnet20k_4e-4_64_0.0001/
mv learning_results.csv resultsResnet20k_4e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 2e-4 -b 64 -lr 0.0001> resultsResnet10k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet10k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet10k_2e-4_64_0.0001/
mv plots/* resultsResnet10k_2e-4_64_0.0001/
mv plots_custom/* resultsResnet10k_2e-4_64_0.0001/
mv learning_results.csv resultsResnet10k_2e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 5e-4 -b 64 -lr 0.0001> resultsResnet10k_5e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet10k_5e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet10k_5e-4_64_0.0001/
mv plots/* resultsResnet10k_5e-4_64_0.0001/
mv plots_custom/* resultsResnet10k_5e-4_64_0.0001/
mv learning_results.csv resultsResnet10k_5e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 7e-4 -b 64 -lr 0.0001> resultsResnet10k_7e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet10k_7e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet10k_7e-4_64_0.0001/
mv plots/* resultsResnet10k_7e-4_64_0.0001/
mv plots_custom/* resultsResnet10k_7e-4_64_0.0001/
mv learning_results.csv resultsResnet10k_7e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 9e-4 -b 64 -lr 0.0001> resultsResnet10k_9e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet10k_9e-4_64_0.0001/logTest40k.dat
mv models/image* resultsResnet10k_9e-4_64_0.0001/
mv plots/* resultsResnet10k_9e-4_64_0.0001/
mv plots_custom/* resultsResnet10k_9e-4_64_0.0001/
mv learning_results.csv resultsResnet10k_9e-4_64_0.0001/

python3 main_ddqn.py -n 40000 -e 4e-5 -b 64 -lr 0.0001> resultsResnet40k_4e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet40k_4e-5_64_0.0001/logTest40k.dat
mv models/image* resultsResnet40k_4e-5_64_0.0001/
mv plots/* resultsResnet40k_4e-5_64_0.0001/
mv plots_custom/* resultsResnet40k_4e-5_64_0.0001/
mv learning_results.csv resultsResnet40k_4e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.0001> resultsResnet40k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet40k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsResnet40k_1e-5_64_0.0001/
mv plots/* resultsResnet40k_1e-5_64_0.0001/
mv plots_custom/* resultsResnet40k_1e-5_64_0.0001/
mv learning_results.csv resultsResnet40k_1e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.0001> resultsResnet40k_2e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet40k_2e-5_64_0.0001/logTest40k.dat
mv models/image* resultsResnet40k_2e-5_64_0.0001/
mv plots/* resultsResnet40k_2e-5_64_0.0001/
mv plots_custom/* resultsResnet40k_2e-5_64_0.0001/
mv learning_results.csv resultsResnet40k_2e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 8e-6 -b 64 -lr 0.0001> resultsResnet40k_8e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsResnet40k_8e-6_64_0.0001/logTest40k.dat
mv models/image* resultsResnet40k_8e-6_64_0.0001/
mv plots/* resultsResnet40k_8e-6_64_0.0001/
mv plots_custom/* resultsResnet40k_8e-6_64_0.0001/
mv learning_results.csv resultsResnet40k_8e-6_64_0.0001/