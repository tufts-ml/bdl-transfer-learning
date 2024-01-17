# bdl-transfer-learning

## `main.py` Arguments

TODO: Explain all arguments used in main.py.
* ```dataset_path``` - path to the folder where the dataset will be downloaded to during the first run and loaded from during later runs.
* ```experiments_path``` - path to the folder where the results will be stored.
* ```model_name``` - name of the experiment. Experiments will be saved as "model_name".csv/"model_name".pth into "experiments_path".
* ```save``` - flag to save the model at the end of the training.
* ```tune``` - flag to generate and use validation set. Used for hyperparameter search.
* ```wandb``` - flag to load the results to [Wandb](https://wandb.ai/site). TODO: Make it possible to log i from other accounts.
* ```wandb_project``` - name of the Wandb project where the results will be saved. The results will be saved under "model_name".
* ```batch_size``` - batch size.
* ```lr_0``` - learning rate.
* ```weight_decay``` - weight decay.
* ```random_state``` - random state. Applied to PyTorch and datasets train/validation split.
* ```n``` - number of training samples.
* ```prior_type``` - type of the experiment. nonlearned - standard transfer learning; adapted - isotropic covariance aka [MAP adaptation](https://aclanthology.org/W04-3237.pdf); learned - learned isotropic covariance from [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279).
* ```prior_path``` - path to the folder containing the prior. Priors from [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279) are available [here](https://github.com/hsouri/BayesianTransferLearning).
* ```number_of_samples_prior``` - Number of low-rank covariance terms of the prior. Default value is 5. Only used with ```prior_path```='learned'.
* ```prior_scale``` - parameter for rescaling the prior as described in [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279). Only used with ```prior_path```='learned'.
* ```prior_eps``` - additive term that is added to the variance terms of the covariance matrix. Default value is 0.1. Only used with ```prior_path```='learned'.

## Reproducing Baseline CIFAR-10 Results

### Install Enviroment
See `bdl-transfer-learning.yml`.

### Download Shwartz-Ziv et al. (2022)'s SimCLR Resnet-50 prior/initialization
Shwartz-Ziv et al. (2022)'s pre-trained prior can be found at https://github.com/hsouri/BayesianTransferLearning. Make sure to download `resnet50_ssl_prior` as this is the prior we use in our paper.

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
