import argparse
import os
import ast
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

import wandb
# Importing our custom module(s)
import folds
import optimizers
import losses
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')    
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--dataset_path', help='Path to CIFAR10', required=True, type=str)
    parser.add_argument('--experiments_path', help='Path to save experiments', required=True, type=str)
    parser.add_argument('--learned_prior', action='store_true', default=False, help='Whether or not to use learned prior (default: False)')
    parser.add_argument('--lr_0', default=0.5, help='Initial learning rate (default: 0.5)', type=float)
    parser.add_argument('--model_name', help='Name for the model file', required=True, type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--number_of_samples_prior', default=5, help='Number of low-rank covariance terms of the prior (default: 5)', type=float)
    parser.add_argument('--prior_path', help='Path to saved priors', required=True, type=str)
    parser.add_argument('--prior_scale', default=1e10, help='Scaling factor for the prior (default: 1e10)', type=float)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to wandb')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')

    args = parser.parse_args()
    print(args)
    # Set torch random seed
    torch.manual_seed(args.random_state)
    # Create checkpoints directory
    utils.makedir_if_not_exist(args.experiments_path)
    # Load labels.csv
    df = pd.read_csv(os.path.join(args.dataset_path, 'labels.csv'), index_col='lesion_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    # Subsample data and create folds
    df = df.sample(n=1000, random_state=args.random_state)
    df['Fold'] = folds.create_folds(df, index_name='lesion_id', random_state=args.random_state)
    # Split folds
    train_df, val_df, test_df = folds.split_folds(df)
    # Create datasets
    train_dataset = utils.ImageDataset2D(train_df)
    val_dataset = utils.ImageDataset2D(val_df, mean_and_std=train_dataset.mean_and_std)
    test_dataset = utils.ImageDataset2D(test_df, mean_and_std=train_dataset.mean_and_std)
    # Create dataloaders
    train_loader_shuffled = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=utils.collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=utils.collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if args.wandb:
        wandb_name = '{}'.format(args.model_name)
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        run = wandb.init(
        project = 'HAM10000',
        name = wandb_name,
        config={
            'batch_size': args.batch_size,
            'dataset_path': args.dataset_path,
            'device': device,
            'experiments_path': args.experiments_path,
            'learned_prior': args.learned_prior,
            'lr_0': args.lr_0,
            'model_name': args.model_name,
            'n': args.n,
            'number_of_samples_prior': args.number_of_samples_prior,
            'prior_path': args.prior_path,
            'prior_scale': args.prior_scale,
            'random_state': args.random_state,
            'wandb': args.wandb,
            'weight_decay': args.weight_decay,
        })
    
    label_set = set(np.array(df.label.to_list()).flatten())
    if label_set == {0, 1}: ce = nn.BCEWithLogitsLoss()
    else: ce = nn.CrossEntropyLoss()
    num_labels, num_samples = np.array(df.label.to_list()).shape
    num_heads = num_labels if label_set == {0, 1} else len(np.unique(df.label.to_list()))
        
    # Load model
    checkpoint = torch.load('{}/resnet50_ssl_prior_model.pt'.format(args.prior_path), map_location=torch.device('cpu'))
    model = resnet50() # Define model
    model.fc = torch.nn.Identity() # Get the classification head off
    model.load_state_dict(checkpoint) # Load the pretrained backbone weights
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True) # Put the proper classification head back
    model.to(device)
    
    if args.learned_prior:
        # Load prior parameters
        mean = torch.load('{}/resnet50_ssl_prior_mean.pt'.format(args.prior_path))
        variance = torch.load('{}/resnet50_ssl_prior_variance.pt'.format(args.prior_path))
        cov_factor = torch.load('{}/resnet50_ssl_prior_covmat.pt'.format(args.prior_path))
        prior_eps = 1e-1 # Default from "Pre-Train Your Loss"
        variance = args.prior_scale * variance + prior_eps # Scale the variance
        cov_mat_sqrt = args.prior_scale * (cov_factor[:args.number_of_samples_prior]) # Scale the low rank covariance
        prior_params = {'mean': mean.cpu(), 'variance': variance.cpu(), 'cov_mat_sqr': cov_mat_sqrt.cpu()}
    else:
        # Initialize empty prior_params (prior_params needs to have prior_params.shape)
        prior_params = {'mean': torch.Tensor([0]), 'variance': torch.Tensor([0]), 'cov_mat_sqr': torch.Tensor([0])}
    
    if args.learned_prior:
        criterion = losses.GaussianPriorCELossShifted(ce, prior_params)
    else:
        criterion = losses.CustomCELoss(ce)
 
    # TODO: Need to add option for SGHMC
    steps = int(30000/5) # 30,000 steps 5 chains
    epochs = int(steps*min(args.batch_size, len(train_dataset))/len(train_dataset))
    number_of_batches = len(train_loader)
    T = epochs*number_of_batches # Total number of iterations
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    columns = ['epoch', 'test_auroc', 'test_loss', 'test_nll', 'test_prior', 'train_auroc', 'train_loss', 'train_nll', 'train_prior', 'val_auroc', 'val_loss', 'val_nll', 'val_prior']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(args.epochs):
        
        lrs = utils.train_one_epoch(model, prior_params, criterion, optimizer, scheduler, train_loader_shuffled)
        
        train_loss, train_nll, train_prior, train_auroc = utils.evaluate(model, prior_params, criterion, train_loader)
        val_loss, val_nll, val_prior, val_auroc = utils.evaluate(model, prior_params, criterion, val_loader)
        test_loss, test_nll, test_prior, test_auroc = utils.evaluate(model, prior_params, criterion, test_loader)
                    
        # Append evaluation metrics to DataFrame
        row = [epoch, test_auroc, test_loss, test_nll, test_prior, train_auroc, train_loss, train_nll, train_prior, val_auroc, val_loss, val_nll, val_prior]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        if args.wandb == True:
            wandb.log({
                'epoch': epoch, 
                'test_auroc': test_auroc, 
                'test_loss': test_loss, 
                'test_nll': test_nll, 
                'test_prior': test_prior,
                'train_auroc': train_auroc, 
                'train_loss': train_loss, 
                'train_nll': train_nll, 
                'train_prior': train_prior,
                'val_auroc': val_auroc, 
                'val_loss': val_loss, 
                'val_nll': val_nll, 
                'val_prior': val_prior,
            })
        
        model_path = os.path.join(args.experiments_path, '{}.csv'.format(args.model_name))
        model_history_df.to_csv(model_path)