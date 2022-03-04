#!/bin/sh


mkdir resultsResNet40k_1e-4_64_0.0001_0.1
mkdir resultsResNet40k_1e-4_64_0.0001_0.5
      

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.0001 -g 0.1 > resultsResNet40k_1e-4_64_0.0001_0.1/logTraining40k.dat
python3 bestPass.py > resultsResNet40k_1e-4_64_0.0001_0.1/logTest40k.dat
mv models/image* resultsResNet40k_1e-4_64_0.0001_0.1/
mv plots/* resultsResNet40k_1e-4_64_0.0001_0.1/
mv plots_custom/* resultsResNet40k_1e-4_64_0.0001_0.1/
mv learning_results.csv resultsResNet40k_1e-4_64_0.0001_0.1/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.0001 -g 0.5 > resultsResnet40k_1e-4_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsResnet40k_1e-4_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsResnet40k_1e-4_64_0.0001_0.5/
mv plots/* resultsResnet40k_1e-4_64_0.0001_0.5/
mv plots_custom/* resultsResnet40k_1e-4_64_0.0001_0.5/
mv learning_results.csv resultsResnet40k_1e-4_64_0.0001_0.5/




