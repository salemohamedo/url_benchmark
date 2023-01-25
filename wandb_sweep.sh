#!/bin/bash

#SBATCH --partition=long                                 # Ask for unkillable job
#SBATCH --cpus-per-task=4                                # Ask for 4 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=40G                                        # Ask for 10 GB of RAM
#SBATCH --time=72:00:00                                   # The job will run for 3 hours
#SBATCH -o /network/scratch/o/omar.salemohamed/logs/wandb-sweep-%j.out  # Write the log on scratch

# 1. Load the required modules
module unload python
module load anaconda/3

# 2. Load environment
conda activate urlb

unset CUDA_VISIBLE_DEVICES

# wandb agent omar-s/url_benchmark/8osmn1tn
# wandb agent omar-s/url_benchmark/dpxxw2mx
wandb agent omar-s/url_benchmark/717uodlt