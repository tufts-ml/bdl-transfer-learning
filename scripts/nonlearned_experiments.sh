#!/bin/bash
#SBATCH -n 1
#SBATCH -p hugheslab
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-1

source ~/.bashrc
conda activate bdl-transfer-learning

# Define an array of commands
experiments=(
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_torchvision' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=10_random_state=1001_weight_decay=0.01' --n=10 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10_torchvision' --weight_decay=0.01"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_torchvision' --lr_0=0.0001 --model_name='nonlearned_lr_0=0.0001_n=10_random_state=2001_weight_decay=0.01' --n=10 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_torchvision' --prior_type='nonlearned' --random_state=2001 --wandb --wandb_project='retrained_CIFAR-10_torchvision' --weight_decay=0.01"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate