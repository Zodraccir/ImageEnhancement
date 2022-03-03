#!/bin/sh

mkdir resultsCustom20k_2e-4_64_0.0001_0.0
mkdir resultsCustom20k_1e-4_64_0.0001_0.5
mkdir resultsCustom20k_1e-3_64_0.0001_0.5
mkdir resultsCustom20k_1e-2_64_0.0001_0.5
mkdir resultsCustom40k_1e-4_64_0.0001_0.5
mkdir resultsCustom40k_1e-5_64_0.0001_0.5
mkdir resultsCustom40k_2e-5_64_0.0001_0.5
mkdir resultsCustom40k_5e-5_64_0.0001_0.5


python3 main_ddqn.py -n 20000 -e 2e-4 -b 64 -lr 0.0001 -g 0.0> resultsCustom20k_2e-4_64_0.0001_0.0/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-4_64_0.0001_0.0/logTest40k.dat
mv models/image* resultsCustom20k_2e-4_64_0.0001_0.0/
mv plots/* resultsCustom20k_2e-4_64_0.0001_0.0/
mv plots_custom/* resultsCustom20k_2e-4_64_0.0001_0.0/
mv learning_results.csv resultsCustom20k_2e-4_64_0.0001_0.0/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0001 -g 0.5> resultsCustom20k_1e-4_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_1e-4_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom20k_1e-4_64_0.0001_0.5/
mv plots/* resultsCustom20k_1e-4_64_0.0001_0.5/
mv plots_custom/* resultsCustom20k_1e-4_64_0.0001_0.5/
mv learning_results.csv resultsCustom20k_1e-4_64_0.0001_0.5/

python3 main_ddqn.py -n 20000 -e 1e-3 -b 64 -lr 0.0001 -g 0.5> resultsCustom20k_1e-3_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_1e-3_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom20k_1e-3_64_0.0001_0.5/
mv plots/* resultsCustom20k_1e-3_64_0.0001_0.5/
mv plots_custom/* resultsCustom20k_1e-3_64_0.0001_0.5/
mv learning_results.csv resultsCustom20k_1e-3_64_0.0001_0.5/

python3 main_ddqn.py -n 20000 -e 1e-2 -b 64 -lr 0.0001 -g 0.5> resultsCustom20k_1e-2_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_1e-2_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom20k_1e-2_64_0.0001_0.5/
mv plots/* resultsCustom20k_1e-2_64_0.0001_0.5/
mv plots_custom/* resultsCustom20k_1e-2_64_0.0001_0.5/
mv learning_results.csv resultsCustom20k_1e-2_64_0.0001_0.5/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.0001 -g 0.5> resultsCustom40k_1e-4_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_1e-4_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom40k_1e-4_64_0.0001_0.5/
mv plots/* resultsCustom40k_1e-4_64_0.0001_0.5/
mv plots_custom/* resultsCustom40k_1e-4_64_0.0001_0.5/
mv learning_results.csv resultsCustom40k_1e-4_64_0.0001_0.5/

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.0001 -g 0.5> resultsCustom40k_1e-5_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_1e-5_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom40k_1e-5_64_0.0001_0.5/
mv plots/* resultsCustom40k_1e-5_64_0.0001_0.5/
mv plots_custom/* resultsCustom40k_1e-5_64_0.0001_0.5/
mv learning_results.csv resultsCustom40k_1e-5_64_0.0001_0.5/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.0001 -g 0.5> resultsCustom40k_2e-5_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_2e-5_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom40k_2e-5_64_0.0001_0.5/
mv plots/* resultsCustom40k_2e-5_64_0.0001_0.5/
mv plots_custom/* resultsCustom40k_2e-5_64_0.0001_0.5/
mv learning_results.csv resultsCustom40k_2e-5_64_0.0001_0.5/

python3 main_ddqn.py -n 40000 -e 5e-5 -b 64 -lr 0.0001 -g 0.5> resultsCustom40k_5e-5_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom40k_5e-5_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom40k_5e-5_64_0.0001_0.5/
mv plots/* resultsCustom40k_5e-5_64_0.0001_0.5/
mv plots_custom/* resultsCustom40k_5e-5_64_0.0001_0.5/
mv learning_results.csv resultsCustom40k_5e-5_64_0.0001_0.5/



