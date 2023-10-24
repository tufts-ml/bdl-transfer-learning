#!/bin/bash
#SBATCH -n 1
#SBATCH -p ccgpu
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-5

source ~/.bashrc
conda activate bdl_2022f_env

# Define an array of commands
experiments=(
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=1000_random_state=1001_weight_decay=0.001' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=1000_random_state=2001_weight_decay=0.001' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --random_state=2001 --wandb --weight_decay=0.001"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=1000_random_state=3001_weight_decay=1e-05' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --random_state=3001 --wandb --weight_decay=1e-05"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --learned_prior --lr_0=0.01 --model_name='learned_lr_0=0.01_n=1000_prior_scale=100000000.0_random_state=1001_weight_decay=0.001' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=100000000.0 --random_state=1001 --wandb --weight_decay=0.001"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --learned_prior --lr_0=0.1 --model_name='learned_lr_0=0.1_n=1000_prior_scale=100.0_random_state=2001_weight_decay=1e-05' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=100.0 --random_state=2001 --wandb --weight_decay=1e-05"
    "python ../src/CIFAR10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10' --learned_prior --lr_0=0.01 --model_name='learned_lr_0=0.01_n=1000_prior_scale=10.0_random_state=3001_weight_decay=1e-05' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=10.0 --random_state=3001 --wandb --weight_decay=1e-05"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate