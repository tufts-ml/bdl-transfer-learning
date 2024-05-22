## Example `main.py`

```
usage: main.py [-h] [--batch_size BATCH_SIZE] [--bb_weight_decay BB_WEIGHT_DECAY]
               [--clf_weight_decay CLF_WEIGHT_DECAY] [--dataset_path DATASET_PATH]
               [--experiments_path EXPERIMENTS_PATH] [--k K] [--lr_0 LR_0]
               [--model_name MODEL_NAME] [--n N] [--num_workers NUM_WORKERS]
               [--prior_eps PRIOR_EPS] [--prior_path PRIOR_PATH] [--prior_type PRIOR_TYPE]
               [--save] [--tune] [--random_state RANDOM_STATE] [--wandb]
               [--wandb_project WANDB_PROJECT]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size (default: 128)
  --bb_weight_decay BB_WEIGHT_DECAY
                        Backbone weight decay (default: 1e-2)
  --clf_weight_decay CLF_WEIGHT_DECAY
                        Classifier weight decay (default: 1e-2)
  --dataset_path DATASET_PATH
                        Path to dataset (default: '')
  --experiments_path EXPERIMENTS_PATH
                        Path to save experiments (default: '')
  --k K                 Rank of low-rank covariance matrix (default: 5)
  --lr_0 LR_0           Initial learning rate (default: 0.5)
  --model_name MODEL_NAME
                        Model name (default: 'test')
  --n N                 Number of training samples (default: 1000)
  --num_workers NUM_WORKERS
                        Number of workers (default: 1)
  --prior_eps PRIOR_EPS
                        Added to prior variance (default: 1e-1)
  --prior_path PRIOR_PATH
                        Path to saved priors (default: '')
  --prior_type PRIOR_TYPE
                        Determines criterion (default: StdPrior)
  --save                Whether or not to save the model (default: False)
  --tune                Whether validation or test set is used (default: False)
  --random_state RANDOM_STATE
                        Random state (default: 42)
  --wandb               Whether or not to log to wandb
  --wandb_project WANDB_PROJECT
                        Wandb project name (default: 'test')
```
