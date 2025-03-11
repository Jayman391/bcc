#!/bin/bash

#####  Slurm preamble
# Job name is descriptive and optional

#SBATCH --job-name=llama_label
#SBATCH --partition=general

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --time=15:00:00

python3 src/analytics/evaluate_predictions.py