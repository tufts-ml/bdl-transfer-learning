# bdl-transfer-learning

## Reproducing Baseline Results

|    | prior_type   |     n |   lr_0 |   random_state |   weight_decay |
|---:|:-------------|------:|-------:|---------------:|---------------:|
|  0 | nonlearned   |    10 | 0.001  |           1001 |         0.01   |
|  1 | nonlearned   |   100 | 0.001  |           1001 |         0.0001 |
|  2 | nonlearned   |  1000 | 0.0001 |           1001 |         0.0001 |
|  3 | nonlearned   | 10000 | 0.01   |           1001 |         0.001  |
|  4 | nonlearned   | 50000 | 0.01   |           1001 |         1e-05  |

`python ../src/CIFAR-10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.001 --model_name='nonlearned_lr_0=0.001_n=10_random_state=1001_weight_decay=0.01' --n=10 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10' --weight_decay=0.01`

`python ../src/CIFAR-10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.001 --model_name='nonlearned_lr_0=0.001_n=100_random_state=1001_weight_decay=0.0001' --n=100 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10' --weight_decay=0.0001`

`python ../src/CIFAR-10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.0001 --model_name='nonlearned_lr_0=0.0001_n=1000_random_state=1001_weight_decay=0.0001' --n=1000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10' --weight_decay=0.0001`

`python ../src/CIFAR-10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=10000_random_state=1001_weight_decay=0.001' --n=10000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10' --weight_decay=0.001`

`python ../src/CIFAR-10_main.py --dataset_path='/cluster/tufts/hugheslab/eharve06/CIFAR-10' --experiments_path='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1' --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=50000_random_state=1001_weight_decay=1e-05' --n=50000 --prior_path='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior' --prior_type='nonlearned' --random_state=1001 --wandb --wandb_project='retrained_CIFAR-10' --weight_decay=1e-05`

Note: If you would like to use (wandb)[https://wandb.ai/] add `--wandb --wandb_project={project_name}`.
