#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=aries
#SBATCH --time=1-2:34:56
#SBATCH --account=vihaan
#SBATCH --mail-type=all 
#SBATCH --mail-user=vihaanakshaay@ucsb.edu
#SBATCH --output=RCNN_out.txt
#SBATCH --error=RCNN_err.txt

# The rest are your jobs

## Use environment from taurus
conda activate earthai

## Run your job
CUDA_VISIBLE_DEVICES=0 python RCNN_seven_band_train.py