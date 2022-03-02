#!/bin/sh

mkdir resultsCustom20k_8e-5_64_0.0001
mkdir resultsCustom20k_1e-4_64_0.0001
mkdir resultsCustom20k_2e-4_64_0.0001
mkdir resultsCustom20k_4e-4_64_0.0001
mkdir resultsCustom10k_2e-4_64_0.0001
mkdir resultsCustom10k_5e-4_64_0.0001
mkdir resultsCustom10k_7e-4_64_0.0001
mkdir resultsCustom10k_9e-4_64_0.0001
mkdir resultsCustom40k_4e-5_64_0.0001
mkdir resultsCustom40k_1e-5_64_0.0001
mkdir resultsCustom40k_2e-5_64_0.0001
mkdir resultsCustom40k_8e-6_64_0.0001


python3 main_ddqn.py -n 20000 -e 8e-5 -b 64 -lr 0.0001> resultsCustom20k_8e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_8e-5_64_0.0001/logTest40k.dat
mv models/image* resultsCustom20k_8e-5_64_0.0001/
mv plots/* resultsCustom20k_8e-5_64_0.0001/
mv plots_custom/* resultsCustom20k_8e-5_64_0.0001/
mv learning_results.csv resultsCustom20k_8e-5_64_0.0001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0001> resultsCustom20k_1e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_1e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom20k_1e-4_64_0.0001/
mv plots/* resultsCustom20k_1e-4_64_0.0001/
mv plots_custom/* resultsCustom20k_1e-4_64_0.0001/
mv learning_results.csv resultsCustom20k_1e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-4 -b 64 -lr 0.0001> resultsCustom20k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom20k_2e-4_64_0.0001/
mv plots/* resultsCustom20k_2e-4_64_0.0001/
mv plots_custom/* resultsCustom20k_2e-4_64_0.0001/
mv learning_results.csv resultsCustom20k_2e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 4e-4 -b 64 -lr 0.0001> resultsCustom20k_4e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_4e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom20k_4e-4_64_0.0001/
mv plots/* resultsCustom20k_4e-4_64_0.0001/
mv plots_custom/* resultsCustom20k_4e-4_64_0.0001/
mv learning_results.csv resultsCustom20k_4e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 2e-4 -b 64 -lr 0.0001> resultsCustom10k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom10k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom10k_2e-4_64_0.0001/
mv plots/* resultsCustom10k_2e-4_64_0.0001/
mv plots_custom/* resultsCustom10k_2e-4_64_0.0001/
mv learning_results.csv resultsCustom10k_2e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 5e-4 -b 64 -lr 0.0001> resultsCustom10k_5e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom10k_5e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom10k_5e-4_64_0.0001/
mv plots/* resultsCustom10k_5e-4_64_0.0001/
mv plots_custom/* resultsCustom10k_5e-4_64_0.0001/
mv learning_results.csv resultsCustom10k_5e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 7e-4 -b 64 -lr 0.0001> resultsCustom10k_7e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom10k_7e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom10k_7e-4_64_0.0001/
mv plots/* resultsCustom10k_7e-4_64_0.0001/
mv plots_custom/* resultsCustom10k_7e-4_64_0.0001/
mv learning_results.csv resultsCustom10k_7e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 9e-4 -b 64 -lr 0.0001> resultsCustom10k_9e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom10k_9e-4_64_0.0001/logTest40k.dat
mv models/image* resultsCustom10k_9e-4_64_0.0001/
mv plots/* resultsCustom10k_9e-4_64_0.0001/
mv plots_custom/* resultsCustom10k_9e-4_64_0.0001/
mv learning_results.csv resultsCustom10k_9e-4_64_0.0001/

python3 main_ddqn.py -n 40000 -e 4e-5 -b 64 -lr 0.0001> resultsCustom40k_4e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_4e-5_64_0.0001/logTest40k.dat
mv models/image* resultsCustom40k_4e-5_64_0.0001/
mv plots/* resultsCustom40k_4e-5_64_0.0001/
mv plots_custom/* resultsCustom40k_4e-5_64_0.0001/
mv learning_results.csv resultsCustom40k_4e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.0001> resultsCustom40k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsCustom40k_1e-5_64_0.0001/
mv plots/* resultsCustom40k_1e-5_64_0.0001/
mv plots_custom/* resultsCustom40k_1e-5_64_0.0001/
mv learning_results.csv resultsCustom40k_1e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.0001> resultsCustom40k_2e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_2e-5_64_0.0001/logTest40k.dat
mv models/image* resultsCustom40k_2e-5_64_0.0001/
mv plots/* resultsCustom40k_2e-5_64_0.0001/
mv plots_custom/* resultsCustom40k_2e-5_64_0.0001/
mv learning_results.csv resultsCustom40k_2e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 8e-6 -b 64 -lr 0.0001> resultsCustom40k_8e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_8e-6_64_0.0001/logTest40k.dat
mv models/image* resultsCustom40k_8e-6_64_0.0001/
mv plots/* resultsCustom40k_8e-6_64_0.0001/
mv plots_custom/* resultsCustom40k_8e-6_64_0.0001/
mv learning_results.csv resultsCustom40k_8e-6_64_0.0001/