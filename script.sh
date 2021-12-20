#!/bin/sh

mkdir resultsAlex40k_1e-5_512_0.002
mkdir resultsAlex40k_5e-5_512_0.002
mkdir resultsAlex40k_2e-5_512_0.002
mkdir resultsAlex40k_1e-4_512_0.002

python3 main_ddqn.py -n 40000 -e 1e-5 -b 512 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-5_512_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-5_512_0.002/
mv plots/* resultsAlex40k_1e-5_512_0.002/
mv plots_custom/* resultsAlex40k_1e-5_512_0.002/
mv results.csv resultsAlex40k_1e-5_512_0.002/


python3 main_ddqn.py -n 40000 -e 5e-5 -b 512 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_5e-5_512_0.002/logTest40k.dat
mv models/image* resultsAlex40k_5e-5_512_0.002/
mv plots/* resultsAlex40k_5e-5_512_0.002/
mv plots_custom/* resultsAlex40k_5e-5_512_0.002/
mv results.csv resultsAlex40k_5e-5_512_0.002/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 512 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_2e-5_512_0.002/logTest40k.dat
mv models/image* resultsAlex40k_2e-5_512_0.002/
mv plots/* resultsAlex40k_2e-5_512_0.002/
mv plots_custom/* resultsAlex40k_2e-5_512_0.002/
mv results.csv resultsAlex40k_2e-5_512_0.002/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 512 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-4_512_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-4_512_0.002/
mv plots/* resultsAlex40k_1e-4_512_0.002/
mv plots_custom/* resultsAlex40k_1e-4_512_0.002/
mv results.csv resultsAlex40k_1e-4_512_0.002/


mkdir resultsAlex40k_1e-5_256_0.002
mkdir resultsAlex40k_5e-5_256_0.002
mkdir resultsAlex40k_2e-5_256_0.002
mkdir resultsAlex40k_1e-4_256_0.002

python3 main_ddqn.py -n 40000 -e 1e-5 -b 256 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-5_256_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-5_256_0.002/
mv plots/* resultsAlex40k_1e-5_256_0.002/
mv plots_custom/* resultsAlex40k_1e-5_256_0.002/
mv results.csv resultsAlex40k_1e-5_256_0.002/


python3 main_ddqn.py -n 40000 -e 5e-5 -b 256 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_5e-5_256_0.002/logTest40k.dat
mv models/image* resultsAlex40k_5e-5_256_0.002/
mv plots/* resultsAlex40k_5e-5_256_0.002/
mv plots_custom/* resultsAlex40k_5e-5_256_0.002/
mv results.csv resultsAlex40k_5e-5_256_0.002/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 256 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_2e-5_256_0.002/logTest40k.dat
mv models/image* resultsAlex40k_2e-5_256_0.002/
mv plots/* resultsAlex40k_2e-5_256_0.002/
mv plots_custom/* resultsAlex40k_2e-5_256_0.002/
mv results.csv resultsAlex40k_2e-5_256_0.002/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 256 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-4_256_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-4_256_0.002/
mv plots/* resultsAlex40k_1e-4_256_0.002/
mv plots_custom/* resultsAlex40k_1e-4_256_0.002/
mv results.csv resultsAlex40k_1e-4_256_0.002/

mkdir resultsAlex40k_1e-5_64_0.002
mkdir resultsAlex40k_5e-5_64_0.002
mkdir resultsAlex40k_2e-5_64_0.002
mkdir resultsAlex40k_1e-4_64_0.002

python3 main_ddqn.py -n 40000 -e 1e-5 -b 64 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-5_64_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-5_64_0.002/
mv plots/* resultsAlex40k_1e-5_64_0.002/
mv plots_custom/* resultsAlex40k_1e-5_64_0.002/
mv results.csv resultsAlex40k_1e-5_64_0.002/


python3 main_ddqn.py -n 40000 -e 5e-5 -b 64 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_5e-5_64_0.002/logTest40k.dat
mv models/image* resultsAlex40k_5e-5_64_0.002/
mv plots/* resultsAlex40k_5e-5_64_0.002/
mv plots_custom/* resultsAlex40k_5e-5_64_0.002/
mv results.csv resultsAlex40k_5e-5_64_0.002/

python3 main_ddqn.py -n 40000 -e 2e-5 -b 64 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_2e-5_64_0.002/logTest40k.dat
mv models/image* resultsAlex40k_2e-5_64_0.002/
mv plots/* resultsAlex40k_2e-5_64_0.002/
mv plots_custom/* resultsAlex40k_2e-5_64_0.002/
mv results.csv resultsAlex40k_2e-5_64_0.002/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.002> logTraining40k.dat
python3 bestPass.py > resultsAlex40k_1e-4_64_0.002/logTest40k.dat
mv models/image* resultsAlex40k_1e-4_64_0.002/
mv plots/* resultsAlex40k_1e-4_64_0.002/
mv plots_custom/* resultsAlex40k_1e-4_64_0.002/
mv results.csv resultsAlex40k_1e-4_64_0.002/