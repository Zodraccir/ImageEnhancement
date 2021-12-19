#!/bin/sh


python3 main_ddqn.py -n 20000 -e 1e-5 -b 64 -lr 0.0005> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-5_64_0.0005/logTest20k.dat
mv models/image* resultsAlex20k_1e-5_64_0.0005/
mv plots/* resultsAlex20k_1e-5_64_0.0005/
mv plots_custom/* resultsAlex20k_1e-5_64_0.0005/
mv results.csv resultsAlex20k_1e-5_64_0.0005/


python3 main_ddqn.py -n 20000 -e 5e-5 -b 64 -lr 0.0005> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_5e-5_64_0.0005/logTest20k.dat
mv models/image* resultsAlex20k_5e-5_64_0.0005/
mv plots/* resultsAlex20k_5e-5_64_0.0005/
mv plots_custom/* resultsAlex20k_5e-5_64_0.0005/
mv results.csv resultsAlex20k_5e-5_64_0.0005/

python3 main_ddqn.py -n 20000 -e 2e-5 -b 64 -lr 0.0005> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_2e-5_64_0.0005/logTest20k.dat
mv models/image* resultsAlex20k_2e-5_64_0.0005/
mv plots/* resultsAlex20k_2e-5_64_0.0005/
mv plots_custom/* resultsAlex20k_2e-5_64_0.0005/
mv results.csv resultsAlex20k_2e-5_64_0.0005/

python3 main_ddqn.py -n 20000 -e 1e-4 -b 64 -lr 0.0005> logTraining20k.dat
python3 bestPass.py > resultsAlex20k_1e-4_64_0.0005/logTest20k.dat
mv models/image* resultsAlex20k_1e-4_64_0.0005/
mv plots/* resultsAlex20k_1e-4_64_0.0005/
mv plots_custom/* resultsAlex20k_1e-4_64_0.0005/
mv results.csv resultsAlex20k_1e-4_64_0.0005/