#!/bin/sh

mkdir resultsResNet20k_5e-6_64_0.001
mkdir resultsResNet20k_5e-6_64_0.0005
mkdir resultsResNet20k_5e-6_256_0.001
mkdir resultsResNet20k_5e-6_256_0.0005

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

python3 main_ddqn.py -n 20000 -e 5e-6 -b 256 -lr 0.001 > resultsResNet20k_5e-6_256_0.001/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_256_0.001/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_256_0.001/
mv plots/* resultsResNet20k_5e-6_256_0.001/
mv plots_custom/* resultsResNet20k_5e-6_256_0.001/
mv results.csv resultsResNet20k_5e-6_256_0.001/
mv learning_results.csv resultsResNet20k_5e-6_256_0.001/

python3 main_ddqn.py -n 20000 -e 5e-6 -b 256 -lr 0.0005 > resultsResNet20k_5e-6_256_0.0005/logTraining20k.dat
python3 bestPass.py > resultsResNet20k_5e-6_256_0.0005/logTest20k.dat
mv models/image* resultsResNet20k_5e-6_256_0.0005/
mv plots/* resultsResNet20k_5e-6_256_0.0005/
mv plots_custom/* resultsResNet20k_5e-6_256_0.0005/
mv results.csv resultsResNet20k_5e-6_256_0.0005/
mv learning_results.csv resultsResNet20k_5e-6_256_0.0005/

mkdir resultsResNet40k_25e-7_64_0.001
mkdir resultsResNet40k_25e-7_64_0.0005
mkdir resultsResNet40k_25e-7_256_0.001
mkdir resultsResNet40k_25e-7_256_0.0005

python3 main_ddqn.py -n 40000 -e 25e-7 -b 64 -lr 0.001 > resultsResNet40k_25e-7_64_0.001/logTraining40k.dat
python3 bestPass.py > resultsResNet40k_25e-7_64_0.001/logTest40k.dat
mv models/image* resultsResNet40k_25e-7_64_0.001/
mv plots/* resultsResNet40k_25e-7_64_0.001/
mv plots_custom/* resultsResNet40k_25e-7_64_0.001/
mv results.csv resultsResNet40k_25e-7_64_0.001/
mv learning_results.csv resultsResNet40k_25e-7_64_0.001/


python3 main_ddqn.py -n 40000 -e 25e-7 -b 64 -lr 0.0005 > resultsResNet40k_25e-7_64_0.0005/logTraining40k.dat
python3 bestPass.py > resultsResNet40k_25e-7_64_0.0005/logTest40k.dat
mv models/image* resultsResNet40k_25e-7_64_0.0005/
mv plots/* resultsResNet40k_25e-7_64_0.0005/
mv plots_custom/* resultsResNet40k_25e-7_64_0.0005/
mv results.csv resultsResNet40k_25e-7_64_0.0005/
mv learning_results.csv resultsResNet40k_25e-7_64_0.0005/


python3 main_ddqn.py -n 40000 -e 25e-7 -b 256 -lr 0.001 > resultsResNet40k_25e-7_256_0.001/logTraining40k.dat
python3 bestPass.py > resultsResNet40k_25e-7_256_0.001/logTest40k.dat
mv models/image* resultsResNet40k_25e-7_256_0.001/
mv plots/* resultsResNet40k_25e-7_256_0.001/
mv plots_custom/* resultsResNet40k_25e-7_256_0.001/
mv results.csv resultsResNet40k_25e-7_256_0.001/
mv learning_results.csv resultsResNet40k_25e-7_256_0.001/

python3 main_ddqn.py -n 40000 -e 25e-7 -b 256 -lr 0.0005 > resultsResNet40k_25e-7_256_0.0005/logTraining40k.dat
python3 bestPass.py > resultsResNet40k_25e-7_256_0.0005/logTest40k.dat
mv models/image* resultsResNet40k_25e-7_256_0.0005/
mv plots/* resultsResNet40k_25e-7_256_0.0005/
mv plots_custom/* resultsResNet40k_25e-7_256_0.0005/
mv results.csv resultsResNet40k_25e-7_256_0.0005/
mv learning_results.csv resultsResNet40k_25e-7_256_0.0005/