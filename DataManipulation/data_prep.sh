#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=2-2:34:56 
#SBATCH --account=vihaan
#SBATCH --mail-type=all 
#SBATCH --mail-user=vihaanakshaay@ucsb.edu 
#SBATCH --output=stdout.txt
#SBATCH --error=stderr.txt

conda activate earthai
python3 prepare_labels_from_shapes.py