#!/bin/sh

mkdir resultsVgg20k_8e-5_64_0.0001
mkdir resultsVgg20k_1e-4_64_0.0001
mkdir resultsVgg20k_2e-4_64_0.0001
mkdir resultsVgg20k_4e-4_64_0.0001
mkdir resultsVgg10k_2e-4_64_0.0001
mkdir resultsVgg10k_5e-4_64_0.0001
mkdir resultsVgg10k_7e-4_64_0.0001
mkdir resultsVgg10k_9e-4_64_0.0001
mkdir resultsVgg40k_4e-5_64_0.0001
mkdir resultsVgg40k_1e-5_64_0.0001
mkdir resultsVgg40k_2e-5_64_0.0001
mkdir resultsVgg40k_8e-6_64_0.0001


python3 main_ddqn.py -n 20000 -e 8e-5 -b 64 -lr 0.0001> resultsVgg20k_8e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg20k_8e-5_64_0.0001/logTest40k.dat
mv models/image* resultsVgg20k_8e-5_64_0.0001/
mv plots/* resultsVgg20k_8e-5_64_0.0001/
mv plots_custom/* resultsVgg20k_8e-5_64_0.0001/
mv learning_results.csv resultsVgg20k_8e-5_64_0.0001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0001> resultsVgg20k_1e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg20k_1e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg20k_1e-4_64_0.0001/
mv plots/* resultsVgg20k_1e-4_64_0.0001/
mv plots_custom/* resultsVgg20k_1e-4_64_0.0001/
mv learning_results.csv resultsVgg20k_1e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 2e-4 -b 64 -lr 0.0001> resultsVgg20k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg20k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg20k_2e-4_64_0.0001/
mv plots/* resultsVgg20k_2e-4_64_0.0001/
mv plots_custom/* resultsVgg20k_2e-4_64_0.0001/
mv learning_results.csv resultsVgg20k_2e-4_64_0.0001/

python3 main_ddqn.py -n 20000 -e 4e-4 -b 64 -lr 0.0001> resultsVgg20k_4e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg20k_4e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg20k_4e-4_64_0.0001/
mv plots/* resultsVgg20k_4e-4_64_0.0001/
mv plots_custom/* resultsVgg20k_4e-4_64_0.0001/
mv learning_results.csv resultsVgg20k_4e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 2e-4 -b 64 -lr 0.0001> resultsVgg10k_2e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg10k_2e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg10k_2e-4_64_0.0001/
mv plots/* resultsVgg10k_2e-4_64_0.0001/
mv plots_custom/* resultsVgg10k_2e-4_64_0.0001/
mv learning_results.csv resultsVgg10k_2e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 5e-4 -b 64 -lr 0.0001> resultsVgg10k_5e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg10k_5e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg10k_5e-4_64_0.0001/
mv plots/* resultsVgg10k_5e-4_64_0.0001/
mv plots_custom/* resultsVgg10k_5e-4_64_0.0001/
mv learning_results.csv resultsVgg10k_5e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 7e-4 -b 64 -lr 0.0001> resultsVgg10k_7e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg10k_7e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg10k_7e-4_64_0.0001/
mv plots/* resultsVgg10k_7e-4_64_0.0001/
mv plots_custom/* resultsVgg10k_7e-4_64_0.0001/
mv learning_results.csv resultsVgg10k_7e-4_64_0.0001/

python3 main_ddqn.py -n 10000 -e 9e-4 -b 64 -lr 0.0001> resultsVgg10k_9e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg10k_9e-4_64_0.0001/logTest40k.dat
mv models/image* resultsVgg10k_9e-4_64_0.0001/
mv plots/* resultsVgg10k_9e-4_64_0.0001/
mv plots_custom/* resultsVgg10k_9e-4_64_0.0001/
mv learning_results.csv resultsVgg10k_9e-4_64_0.0001/

python3 main_ddqn.py -n 40000 -e 4e-5 -b 64 -lr 0.0001> resultsVgg40k_4e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg40k_4e-5_64_0.0001/logTest40k.dat
mv models/image* resultsVgg40k_4e-5_64_0.0001/
mv plots/* resultsVgg40k_4e-5_64_0.0001/
mv plots_custom/* resultsVgg40k_4e-5_64_0.0001/
mv learning_results.csv resultsVgg40k_4e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.0001> resultsVgg40k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg40k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsVgg40k_1e-5_64_0.0001/
mv plots/* resultsVgg40k_1e-5_64_0.0001/
mv plots_custom/* resultsVgg40k_1e-5_64_0.0001/
mv learning_results.csv resultsVgg40k_1e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.0001> resultsVgg40k_2e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg40k_2e-5_64_0.0001/logTest40k.dat
mv models/image* resultsVgg40k_2e-5_64_0.0001/
mv plots/* resultsVgg40k_2e-5_64_0.0001/
mv plots_custom/* resultsVgg40k_2e-5_64_0.0001/
mv learning_results.csv resultsVgg40k_2e-5_64_0.0001/

python3 main_ddqn.py -n 40000 -e 8e-6 -b 64 -lr 0.0001> resultsVgg40k_8e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsVgg40k_8e-6_64_0.0001/logTest40k.dat
mv models/image* resultsVgg40k_8e-6_64_0.0001/
mv plots/* resultsVgg40k_8e-6_64_0.0001/
mv plots_custom/* resultsVgg40k_8e-6_64_0.0001/
mv learning_results.csv resultsVgg40k_8e-6_64_0.0001/