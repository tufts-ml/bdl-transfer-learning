## `main.py` Arguments
* ```dataset_path``` - path to the folder where the dataset will be downloaded to during the first run and loaded from during later runs.
* ```experiments_path``` - path to the folder where the results will be stored.
* ```model_name``` - name of the experiment. Experiments will be saved as "model_name".csv/"model_name".pth into "experiments_path".
* ```save``` - flag to save the model at the end of the training.
* ```tune``` - flag to generate and use validation set. Used for hyperparameter search.
* ```wandb``` - flag to load the results to [Wandb](https://wandb.ai/site). Note: If you would like to use [wandb](https://wandb.ai/) make sure to change `os.environ['WANDB_API_KEY']={your_wandb_api_key}` and add `--wandb --wandb_project={project_name}` to the commands above.
* ```wandb_project``` - name of the Wandb project where the results will be saved. The results will be saved under "model_name".
* ```batch_size``` - batch size.
* ```lr_0``` - learning rate.
* ```weight_decay``` - weight decay.
* ```random_state``` - random state. Applied to PyTorch and datasets train/validation split.
* ```n``` - number of training samples.
* ```prior_type``` - type of the experiment. nonlearned - standard transfer learning; adapted - isotropic covariance aka [MAP adaptation](https://aclanthology.org/W04-3237.pdf); learned - learned isotropic covariance from [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279).
* ```prior_path``` - path to the folder containing the prior. Priors from [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279) are available [here](https://github.com/hsouri/BayesianTransferLearning).
* ```number_of_samples_prior``` - Number of low-rank covariance terms of the prior. Default value is 5. Only used with *prior_path*='learned'.
* ```prior_scale``` - parameter for rescaling the prior as described in [Pre-Train Your Loss](https://arxiv.org/abs/2205.10279). Only used with *prior_path*='learned'.
* ```prior_eps``` - additive term that is added to the variance terms of the covariance matrix. Default value is 0.1. Only used with *prior_path*='learned'.
