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
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 64)')
    parser.add_argument('--dataset_path', type=str, default='/cluster/tufts/hugheslab/eharve06/HAM10000', help='path to dataset (default: "/cluster/tufts/hugheslab/eharve06/HAM10000")')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 200)')
    parser.add_argument('--experiments_path', type=str, default='/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/HAM10000', help='path to save experiments (default: "/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/HAM10000")')
    parser.add_argument('--learned_prior', action='store_true', default=False, help='whether or not to use learned prior')
    parser.add_argument('--lr_0', type=float, default=0.5, help='number of epochs to train (default: 0.5)')
    parser.add_argument('--prior_num_terms', type=float, default=5, help='number of low-rank covariance terms of the prior (default: 5)')
    parser.add_argument('--prior_path', type=str, default='/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior', help='path to saved priors (default: "/cluster/tufts/hugheslab/eharve06/resnet50-ssl-prior")')
    parser.add_argument('--prior_scale', type=float, default=1e10, help='scaling factor fir the prior (default: 1e10)')
    parser.add_argument('--random_state', type=int, default=42, help='random state (default: 42)')
    parser.add_argument('--wandb', action='store_true', default=False, help='whether or not to log to wandb')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay (default: 5e-4)')

    args = parser.parse_args()
    print(args)
    # Set torch random seed
    torch.manual_seed(args.random_state)
    # Create checkpoints directory
    utils.makedir_if_not_exist(args.experiments_path)
    utils.makedir_if_not_exist('{}/random_state={}'.format(args.experiments_path, args.random_state))

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
    # TODO: Entire dataset is loaded to memory. This won't work for 3D datasets.
    train_images, train_labels = zip(*[train_dataset[i] for i in range(len(train_dataset))])
    train_dataset = TensorDataset(torch.stack(train_images), torch.Tensor(train_labels).long())
    val_images, val_labels = zip(*[val_dataset[i] for i in range(len(val_dataset))])
    val_dataset = TensorDataset(torch.stack(val_images), torch.Tensor(val_labels).long())
    test_images, test_labels = zip(*[test_dataset[i] for i in range(len(test_dataset))])
    test_dataset = TensorDataset(torch.stack(test_images), torch.Tensor(test_labels).long())
    # Create dataloaders
    train_loader_shuffled = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True, drop_last=True, collate_fn=utils.collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), collate_fn=utils.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), collate_fn=utils.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=len(test_loader), collate_fn=utils.collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if args.wandb:
        wandb_name = 'learned_prior_scale={}_lr_0={}_weight_decay={}_random_state={}'.format(args.prior_scale, args.lr_0, args.weight_decay, args.random_state) if args.learned_prior else 'nonlearned_lr_0={}_weight_decay={}_random_state={}'.format(args.lr_0, args.weight_decay, args.random_state)
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        run = wandb.init(
        project = 'HAM10000',
        name = wandb_name,
        config={
            'batch_size': args.batch_size,
            'dataset_path': args.dataset_path,
            'device': device,
            'epochs': args.epochs,
            'learning_rate': args.lr_0,
            'prior_path': args.prior_path,
            'prior_num_terms': args.prior_num_terms,
            'prior_scale': args.prior_scale,
            'test_set_size': len(test_dataset),
            'test_loader_size': len(test_loader),
            'training_set_size': len(train_dataset),
            'training_loader_size': len(train_loader),
            'val_set_size': len(val_dataset),
            'val_loader_size': len(val_loader),
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
        prior_eps = 1e-1 # Default from "pretrain your loss"
        variance = args.prior_scale * variance + prior_eps # Scale the variance
        cov_mat_sqrt = args.prior_scale * (cov_factor[:args.prior_num_terms]) # Scale the low rank covariance
        prior_params = {'mean': mean.cpu(), 'variance': variance.cpu(), 'cov_mat_sqr': cov_mat_sqrt.cpu()}
    else:
        # prior_params aren't used but initialized
        prior_params = {'mean': torch.Tensor([0]), 'variance': torch.Tensor([0]), 'cov_mat_sqr': torch.Tensor([0])}
    
    if args.learned_prior:
        criterion = losses.GaussianPriorCELossShifted(ce, prior_params)
    else:
        criterion = losses.CustomCELoss(ce)
 
    # TODO: Need to add option for SGHMC
    num_batch = len(train_loader)
    T = args.epochs*num_batch # Total number of iterations
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, 
                                momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
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
        
        model_path = '{}/random_state={}/learned_prior_scale={}_lr_0={}_weight_decay={}.csv'.format(args.experiments_path, args.random_state, args.prior_scale, args.lr_0, args.weight_decay) if args.learned_prior else '{}/random_state={}/nonlearned_lr_0={}_weight_decay={}.csv'.format(args.experiments_path, args.random_state, args.lr_0, args.weight_decay)
        model_history_df.to_csv(model_path)
