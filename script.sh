#!/bin/sh

mkdir resultsAlex20k_8e-5_64_0.0001
mkdir resultsAlex20k_1e-4_64_0.0001
mkdir resultsAlex20k_2e-4_64_0.0001
mkdir resultsAlex20k_4e-4_64_0.0001
mkdir resultsAlex10k_2e-4_64_0.0001
mkdir resultsAlex10k_5e-4_64_0.0001
mkdir resultsAlex10k_7e-4_64_0.0001
mkdir resultsAlex10k_9e-4_64_0.0001
mkdir resultsAlex40k_4e-5_64_0.0001
mkdir resultsAlex40k_1e-5_64_0.0001
mkdir resultsAlex40k_2e-5_64_0.0001
mkdir resultsAlex40k_8e-6_64_0.0001


python3 main_ddqn.py -n 20000 -e 8e-5 -b 64 -lr 0.0001> resultsAlex20k_8e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_8e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_8e-5_64_0.0001/
mv plots/* resultsAlex20k_8e-5_64_0.0001/
mv plots_custom/* resultsAlex20k_8e-5_64_0.0001/
mv learning_results.csv resultsAlex20k_8e-5_64_0.0001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0001> resultsAlex20k_1e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_1e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_1e-4_64_0.0001/
mv plots/* resultsAlex20k_1e-4_64_0.0001/
mv plots_custom/* resultsAlex20k_1e-4_64_0.0001/
mv learning_results.csv resultsAlex20k_1e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-4 -b 64 -lr 0.0001> resultsAlex20k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_2e-4_64_0.0001/
mv plots/* resultsAlex20k_2e-4_64_0.0001/
mv plots_custom/* resultsAlex20k_2e-4_64_0.0001/
mv learning_results.csv resultsAlex20k_2e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 4e-4 -b 64 -lr 0.0001> resultsAlex20k_4e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_4e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_4e-4_64_0.0001/
mv plots/* resultsAlex20k_4e-4_64_0.0001/
mv plots_custom/* resultsAlex20k_4e-4_64_0.0001/
mv learning_results.csv resultsAlex20k_4e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 2e-4 -b 64 -lr 0.0001> resultsAlex10k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex10k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex10k_2e-4_64_0.0001/
mv plots/* resultsAlex10k_2e-4_64_0.0001/
mv plots_custom/* resultsAlex10k_2e-4_64_0.0001/
mv learning_results.csv resultsAlex10k_2e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 5e-4 -b 64 -lr 0.0001> resultsAlex10k_5e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex10k_5e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex10k_5e-4_64_0.0001/
mv plots/* resultsAlex10k_5e-4_64_0.0001/
mv plots_custom/* resultsAlex10k_5e-4_64_0.0001/
mv learning_results.csv resultsAlex10k_5e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 7e-4 -b 64 -lr 0.0001> resultsAlex10k_7e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex10k_7e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex10k_7e-4_64_0.0001/
mv plots/* resultsAlex10k_7e-4_64_0.0001/
mv plots_custom/* resultsAlex10k_7e-4_64_0.0001/
mv learning_results.csv resultsAlex10k_7e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 9e-4 -b 64 -lr 0.0001> resultsAlex10k_9e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex10k_9e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex10k_9e-4_64_0.0001/
mv plots/* resultsAlex10k_9e-4_64_0.0001/
mv plots_custom/* resultsAlex10k_9e-4_64_0.0001/
mv learning_results.csv resultsAlex10k_9e-4_64_0.0001/

python3 main_ddqn.py -n 40000 -e 4e-5 -b 64 -lr 0.0001> resultsAlex40k_4e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex40k_4e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex40k_4e-5_64_0.0001/
mv plots/* resultsAlex40k_4e-5_64_0.0001/
mv plots_custom/* resultsAlex40k_4e-5_64_0.0001/
mv learning_results.csv resultsAlex40k_4e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.0001> resultsAlex40k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex40k_1e-5_64_0.0001/
mv plots/* resultsAlex40k_1e-5_64_0.0001/
mv plots_custom/* resultsAlex40k_1e-5_64_0.0001/
mv learning_results.csv resultsAlex40k_1e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.0001> resultsAlex40k_2e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex40k_2e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex40k_2e-5_64_0.0001/
mv plots/* resultsAlex40k_2e-5_64_0.0001/
mv plots_custom/* resultsAlex40k_2e-5_64_0.0001/
mv learning_results.csv resultsAlex40k_2e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 8e-6 -b 64 -lr 0.0001> resultsAlex40k_8e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex40k_8e-6_64_0.0001/logTest40k.dat
mv models/image* resultsAlex40k_8e-6_64_0.0001/
mv plots/* resultsAlex40k_8e-6_64_0.0001/
mv plots_custom/* resultsAlex40k_8e-6_64_0.0001/
mv learning_results.csv resultsAlex40k_8e-6_64_0.0001/