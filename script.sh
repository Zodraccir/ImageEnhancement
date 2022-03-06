#!/bin/sh


mkdir resultsAlexNet40k_1e-4_64_0.0001_0.1
mkdir resultsAlexNet40k_1e-4_64_0.0001_0.5
      

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.0001 -g 0.1 > resultsAlexNet40k_1e-4_64_0.0001_0.1/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-4_64_0.0001_0.1/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-4_64_0.0001_0.1/
mv plots/* resultsAlexNet40k_1e-4_64_0.0001_0.1/
mv plots_custom/* resultsAlexNet40k_1e-4_64_0.0001_0.1/
mv learning_results.csv resultsAlexNet40k_1e-4_64_0.0001_0.1/

python3 main_ddqn.py -n 40000 -e 1e-4 -b 64 -lr 0.0001 -g 0.5 > resultsAlexNet40k_1e-4_64_0.0001_0.5/logTraining40k.dat
python3 bestPass.py > resultsAlexNet40k_1e-4_64_0.0001_0.5/logTest40k.dat
mv models/image* resultsAlexNet40k_1e-4_64_0.0001_0.5/
mv plots/* resultsAlexNet40k_1e-4_64_0.0001_0.5/
mv plots_custom/* resultsAlexNet40k_1e-4_64_0.0001_0.5/
mv learning_results.csv resultsAlexNet40k_1e-4_64_0.0001_0.5/




