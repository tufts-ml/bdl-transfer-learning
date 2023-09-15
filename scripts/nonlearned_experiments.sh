#!/bin/bash
#SBATCH -n 1
#SBATCH -p ccgpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-23%8

source ~/.bashrc
conda activate bdl_2022f_env

# Define an array of commands
experiments=(
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=0.01"
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=0.0001"
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=1e-05"
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=1e-06"
    "python ../src/main.py --lr_0=0.1 --random_state=1001 --wandb --weight_decay=0.0"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=0.01"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=0.0001"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=1e-05"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=1e-06"
    "python ../src/main.py --lr_0=0.01 --random_state=1001 --wandb --weight_decay=0.0"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=0.01"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=0.0001"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=1e-05"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=1e-06"
    "python ../src/main.py --lr_0=0.001 --random_state=1001 --wandb --weight_decay=0.0"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=0.01"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=0.0001"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=1e-05"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=1e-06"
    "python ../src/main.py --lr_0=0.0001 --random_state=1001 --wandb --weight_decay=0.0"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate