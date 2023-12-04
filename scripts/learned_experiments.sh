#!/bin/bash
#SBATCH -n 1
#SBATCH -p preempt
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=64g
#SBATCH -o /cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out # Write stdout to file named log_JOBIDNUM.out in log dir
#SBATCH -e /cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err # Write stderr to file named log_JOBIDNUM.err in log dir
#SBATCH --array=0-5

source ~/.bashrc
conda activate bdl_2022f_env

# Define an array of commands
experiments=(
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.0001 --model_name='learned_lr_0=0.0001_n=100_prior_scale=100000000.0_random_state=1001_weight_decay=0.01' --n=100 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=100000000.0 --prior_type='learned' --random_state=1001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.01"
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.01 --model_name='learned_lr_0=0.01_n=100_prior_scale=10.0_random_state=2001_weight_decay=0.001' --n=100 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=10.0 --prior_type='learned' --random_state=2001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.001"
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.0001 --model_name='learned_lr_0=0.0001_n=100_prior_scale=1.0_random_state=3001_weight_decay=0.0001' --n=100 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=1.0 --prior_type='learned' --random_state=3001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.0001"
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.01 --model_name='learned_lr_0=0.01_n=1000_prior_scale=1000000.0_random_state=1001_weight_decay=0.01' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=1000000.0 --prior_type='learned' --random_state=1001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.01"
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.0001 --model_name='learned_lr_0=0.0001_n=1000_prior_scale=100.0_random_state=2001_weight_decay=0.01' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=100.0 --prior_type='learned' --random_state=2001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.01"
    "python ../src/HAM10000_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/HAM10000' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_HAM10000' --lr_0=0.01 --model_name='learned_lr_0=0.01_n=1000_prior_scale=100.0_random_state=3001_weight_decay=0.01' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_scale=100.0 --prior_type='learned' --random_state=3001 --wandb --wandb_project='retrained_HAM10000' --weight_decay=0.01"
)
    
eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate