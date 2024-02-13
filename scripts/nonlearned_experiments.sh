#!/bin/bash
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -o /log_%j.out # Write stdout to file named log_JOBIDNUM.out
#SBATCH -e /log_%j.err # Write stderr to file named log_JOBIDNUM.err
#SBATCH --array=0-n # TODO: Change n to the number of experiments.

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
# TODO: See notebooks for generating scripts.
)
    
eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate