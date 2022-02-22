#!/bin/sh

mkdir resultsAlex20k_6e-5_64_0.0001
mkdir resultsAlex10k_2e-5_64_0.0001
mkdir resultsAlex15k_5e-5_64_0.0001
mkdir resultsAlex25k_1e-5_64_0.0001
mkdir resultsAlex30k_8e-6_64_0.0001
mkdir resultsAlex40k_6e-6_64_0.0001


python3 main_ddqn.py -n 20000 -e 6e-5 -b 64 -lr 0.0001> resultsAlex20k_6e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex20k_6e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex20k_6e-5_64_0.0001/
mv plots/* resultsAlex20k_6e-5_64_0.0001/
mv plots_custom/* resultsAlex20k_6e-5_64_0.0001/
mv results.csv resultsAlex20k_6e-5_64_0.0001/
mv learning_results.csv resultsAlex20k_6e-5_64_0.0001/

python3 main_ddqn.py -n 10000 -e 1e-4 -b 64 -lr 0.0001 > resultsAlex10k_1e-4_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex10k_1e-4_64_0.0001/logTest40k.dat
mv models/image* resultsAlex10k_1e-4_64_0.0001/
mv plots/* resultsAlex10k_1e-4_64_0.0001/
mv plots_custom/* resultsAlex10k_1e-4_64_0.0001/
mv results.csv resultsAlex10k_1e-4_64_0.0001/
mv learning_results.csv resultsAlex10k_1e-4_64_0.0001/


python3 main_ddqn.py -n 15000 -e 5e-5 -b 64 -lr 0.0001> resultsAlex15k_5e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex15k_5e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex15k_5e-5_64_0.0001/
mv plots/* resultsAlex15k_5e-5_64_0.0001/
mv plots_custom/* resultsAlex15k_5e-5_64_0.0001/
mv results.csv resultsAlex15k_5e-5_64_0.0001/
mv learning_results.csv resultsAlex15k_5e-5_64_0.0001/

python3 main_ddqn.py -n 25000 -e 1e-5 -b 64 -lr 0.0001> resultsAlex25k_1e-5_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex25k_1e-5_64_0.0001/logTest40k.dat
mv models/image* resultsAlex25k_1e-5_64_0.0001/
mv plots/* resultsAlex25k_1e-5_64_0.0001/
mv plots_custom/* resultsAlex25k_1e-5_64_0.0001/
mv results.csv resultsAlex25k_1e-5_64_0.0001/
mv learning_results.csv resultsAlex25k_1e-5_64_0.0001/

python3 main_ddqn.py -n 30000 -e 8e-6 -b 64 -lr 0.0001> resultsAlex30k_8e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex30k_8e-6_64_0.0001/logTest40k.dat
mv models/image* resultsAlex30k_8e-6_64_0.0001/
mv plots/* resultsAlex30k_8e-6_64_0.0001/
mv plots_custom/* resultsAlex30k_8e-6_64_0.0001/
mv results.csv resultsAlex30k_8e-6_64_0.0001/
mv learning_results.csv resultsAlex30k_8e-6_64_0.0001/

python3 main_ddqn.py -n 40000 -e 6e-6 -b 64 -lr 0.0001> resultsAlex40k_6e-6_64_0.0001/logTraining40k.dat
python3 bestPass.py > resultsAlex40k_6e-6_64_0.0001/logTest40k.dat
mv models/image* resultsAlex40k_6e-6_64_0.0001/
mv plots/* resultsAlex40k_6e-6_64_0.0001/
mv plots_custom/* resultsAlex40k_6e-6_64_0.0001/
mv results.csv resultsAlex40k_6e-6_64_0.0001/
mv learning_results.csv resultsAlex40k_6e-6_64_0.0001/


