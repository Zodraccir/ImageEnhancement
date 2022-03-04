#!/bin/sh


mkdir resultsCustom20k_2e-5_64_0.0001_0.1
mkdir resultsCustom20k_2e-5_64_0.0001_0.5
mkdir resultsCustom20k_2e-5_64_0.0001_0.9
mkdir resultsCustom20k_2e-5_64_0.0001_0.99


python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0001 -g 0.1> resultsCustom20k_2e-5_64_0.0001_0.1/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-5_64_0.0001_0.1/logTest40k.dat
mv models/image* resultsCustom20k_2e-5_64_0.0001_0.1/
mv plots/* resultsCustom20k_2e-5_64_0.0001_0.1/
mv plots_custom/* resultsCustom20k_2e-5_64_0.0001_0.1/
mv learning_results.csv resultsCustom20k_2e-5_64_0.0001_0.1/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0001 -g 0.5> resultsCustom20k_2e-5_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-5_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsCustom20k_2e-5_64_0.0001_0.5/
mv plots/* resultsCustom20k_2e-5_64_0.0001_0.5/
mv plots_custom/* resultsCustom20k_2e-5_64_0.0001_0.5/
mv learning_results.csv resultsCustom20k_2e-5_64_0.0001_0.5/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0001 -g 0.9> resultsCustom20k_2e-5_64_0.0001_0.9/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-5_64_0.0001_0.9/logTest40k.dat
mv models/image* resultsCustom20k_2e-5_64_0.0001_0.9/
mv plots/* resultsCustom20k_2e-5_64_0.0001_0.9/
mv plots_custom/* resultsCustom20k_2e-5_64_0.0001_0.9/
mv learning_results.csv resultsCustom20k_2e-5_64_0.0001_0.9/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0001 -g 0.99> resultsCustom20k_2e-5_64_0.0001_0.99/logTraining40k.dat
python3 bestPass.py > resultsCustom20k_2e-5_64_0.0001_0.99/logTest40k.dat
mv models/image* resultsCustom20k_2e-5_64_0.0001_0.99/
mv plots/* resultsCustom20k_2e-5_64_0.0001_0.99/
mv plots_custom/* resultsCustom20k_2e-5_64_0.0001_0.99/
mv learning_results.csv resultsCustom20k_2e-5_64_0.0001_0.99/



