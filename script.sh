#!/bin/sh

mkdir resultsAlex20k_1e-5_512_0.001
mkdir resultsAlex20k_5e-5_512_0.001
mkdir resultsAlex20k_2e-5_512_0.001
mkdir resultsAlex20k_1e-4_512_0.001

python3 main_ddqn.py -n 20000 -e 1e-5 -b 512 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-5_512_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-5_512_0.001/
mv plots/* resultsAlex20k_1e-5_512_0.001/
mv plots_custom/* resultsAlex20k_1e-5_512_0.001/
mv results.csv resultsAlex20k_1e-5_512_0.001/


python3 main_ddqn.py -n 20000 -e 5e-5 -b 512 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_5e-5_512_0.001/logTest20k.dat
mv models/image* resultsAlex20k_5e-5_512_0.001/
mv plots/* resultsAlex20k_5e-5_512_0.001/
mv plots_custom/* resultsAlex20k_5e-5_512_0.001/
mv results.csv resultsAlex20k_5e-5_512_0.001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 512 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_2e-5_512_0.001/logTest20k.dat
mv models/image* resultsAlex20k_2e-5_512_0.001/
mv plots/* resultsAlex20k_2e-5_512_0.001/
mv plots_custom/* resultsAlex20k_2e-5_512_0.001/
mv results.csv resultsAlex20k_2e-5_512_0.001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 512 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-4_512_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-4_512_0.001/
mv plots/* resultsAlex20k_1e-4_512_0.001/
mv plots_custom/* resultsAlex20k_1e-4_512_0.001/
mv results.csv resultsAlex20k_1e-4_512_0.001/


mkdir resultsAlex20k_1e-5_256_0.001
mkdir resultsAlex20k_5e-5_256_0.001
mkdir resultsAlex20k_2e-5_256_0.001
mkdir resultsAlex20k_1e-4_256_0.001

python3 main_ddqn.py -n 20000 -e 1e-5 -b 256 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-5_256_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-5_256_0.001/
mv plots/* resultsAlex20k_1e-5_256_0.001/
mv plots_custom/* resultsAlex20k_1e-5_256_0.001/
mv results.csv resultsAlex20k_1e-5_256_0.001/


python3 main_ddqn.py -n 20000 -e 5e-5 -b 256 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_5e-5_256_0.001/logTest20k.dat
mv models/image* resultsAlex20k_5e-5_256_0.001/
mv plots/* resultsAlex20k_5e-5_256_0.001/
mv plots_custom/* resultsAlex20k_5e-5_256_0.001/
mv results.csv resultsAlex20k_5e-5_256_0.001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 256 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_2e-5_256_0.001/logTest20k.dat
mv models/image* resultsAlex20k_2e-5_256_0.001/
mv plots/* resultsAlex20k_2e-5_256_0.001/
mv plots_custom/* resultsAlex20k_2e-5_256_0.001/
mv results.csv resultsAlex20k_2e-5_256_0.001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 256 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-4_256_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-4_256_0.001/
mv plots/* resultsAlex20k_1e-4_256_0.001/
mv plots_custom/* resultsAlex20k_1e-4_256_0.001/
mv results.csv resultsAlex20k_1e-4_256_0.001/

mkdir resultsAlex20k_1e-5_64_0.001
mkdir resultsAlex20k_5e-5_64_0.001
mkdir resultsAlex20k_2e-5_64_0.001
mkdir resultsAlex20k_1e-4_64_0.001

python3 main_ddqn.py -n 20000 -e 1e-5 -b 64 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-5_64_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-5_64_0.001/
mv plots/* resultsAlex20k_1e-5_64_0.001/
mv plots_custom/* resultsAlex20k_1e-5_64_0.001/
mv results.csv resultsAlex20k_1e-5_64_0.001/


python3 main_ddqn.py -n 20000 -e 5e-5 -b 64 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_5e-5_64_0.001/logTest20k.dat
mv models/image* resultsAlex20k_5e-5_64_0.001/
mv plots/* resultsAlex20k_5e-5_64_0.001/
mv plots_custom/* resultsAlex20k_5e-5_64_0.001/
mv results.csv resultsAlex20k_5e-5_64_0.001/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.001/logTest20k.dat
mv models/image* resultsAlex20k_2e-5_64_0.001/
mv plots/* resultsAlex20k_2e-5_64_0.001/
mv plots_custom/* resultsAlex20k_2e-5_64_0.001/
mv results.csv resultsAlex20k_2e-5_64_0.001/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.001> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-4_64_0.001/logTest20k.dat
mv models/image* resultsAlex20k_1e-4_64_0.001/
mv plots/* resultsAlex20k_1e-4_64_0.001/
mv plots_custom/* resultsAlex20k_1e-4_64_0.001/
mv results.csv resultsAlex20k_1e-4_64_0.001/