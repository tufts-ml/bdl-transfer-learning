#!/bin/bash
#SBATCH -n 1
#SBATCH -p hugheslab
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-0

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.001 --model_name='learned_lr_0=0.001_n=100_prior_scale=1000000000.0_random_state=3001_weight_decay=0.0' --n=100 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=1000000000.0 --prior_type='learned' --random_state=3001 --wandb --wandb_project='retrained_CIFAR-10_Copy1' --weight_decay=0.0"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate