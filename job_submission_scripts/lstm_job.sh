#!/bin/bash --login

#SBATCH --time=24:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=50G

conda activate tf

python /mnt/home/tairaeli/cse404_project/machine-learning-optiver/LSTM.py