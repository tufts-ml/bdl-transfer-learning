# bdl-transfer-learning

## Reproducing Baseline CIFAR-10 Results

### Install Enviroment
See `bdl-transfer-learning.yml`.

### Download Shwartz-Ziv et al. (2022)'s SimCLR Resnet-50 prior/initialization
Shwartz-Ziv et al. (2022)'s pre-trained prior can be found at https://github.com/hsouri/BayesianTransferLearning. In our paper, we use `resnet50_ssl_prior`.

### Optimal Hyperparameters

Optimal hyperparameters for standard transfer learning with the specified random state.

|    | prior_type   |     n |   lr_0 |   random_state |   weight_decay |
|---:|:-------------|------:|-------:|---------------:|---------------:|
|  0 | nonlearned   |    10 | 0.01   |           1001 |         0.01   |
|  1 | nonlearned   |   100 | 0.01   |           1001 |         0.0    |
|  2 | nonlearned   |  1000 | 0.0001 |           1001 |         0.0001 |
|  3 | nonlearned   | 10000 | 0.01   |           1001 |         0.001  |
|  4 | nonlearned   | 50000 | 0.01   |           1001 |         0.001  |

### Command Line Input

To evaluate the optimal hyperparameters for standard transfer learning, use the following commands with the specified random state:

`python CIFAR-10_main.py --dataset_path={dataset_path} --experiments_path={experiments_path} --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=10_random_state=1001_weight_decay=0.01' --n=10 --prior_path={prior_path} --prior_type='nonlearned' --random_state=1001 --weight_decay=0.01`

`python CIFAR-10_main.py --dataset_path={dataset_path} --experiments_path={experiments_path} --lr_0=0.001 --model_name='nonlearned_lr_0=0.01_n=100_random_state=1001_weight_decay=0.0' --n=100 --prior_path={prior_path} --prior_type='nonlearned' --random_state=1001 --weight_decay=0.0`

`python CIFAR-10_main.py --dataset_path={dataset_path} --experiments_path={experiments_path} --lr_0=0.0001 --model_name='nonlearned_lr_0=0.0001_n=1000_random_state=1001_weight_decay=0.0001' --n=1000 --prior_path={prior_path} --prior_type='nonlearned' --random_state=1001 --weight_decay=0.0001`

`python CIFAR-10_main.py --dataset_path={dataset_path} --experiments_path={experiments_path} --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=10000_random_state=1001_weight_decay=0.001' --n=10000 --prior_path={prior_path} --prior_type='nonlearned' --random_state=1001 --weight_decay=0.001`

`python CIFAR-10_main.py --dataset_path={dataset_path} --experiments_path={experiments_path} --lr_0=0.01 --model_name='nonlearned_lr_0=0.01_n=50000_random_state=1001_weight_decay=0.001' --n=50000 --prior_path={prior_path} --prior_type='nonlearned' --random_state=1001 --weight_decay=0.001`

Note: If you would like to use [wandb](https://wandb.ai/) make sure to change `os.environ['WANDB_API_KEY']={your_wandb_api_key}` and add `--wandb --wandb_project={project_name}` to the commands above.

### Visualizing Results

To visualize results see `notebooks/demo.ipynb`.
