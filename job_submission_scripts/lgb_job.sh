#!/bin/bash --login

#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=50G

conda activate data_sc

python /mnt/home/tairaeli/cse404_project/machine-learning-optiver/lightgbm_gridsearch.py