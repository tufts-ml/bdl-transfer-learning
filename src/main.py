import argparse
import os
import ast
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

from evaluation_metrics import *
from folds import *
from losses import *
from utils import *

# python bdl-transfer-learning/src/main.py --checkpoints_dir='/cluster/home/eharve06/bdl-transfer-learning/checkpoints' --prior_dir='/cluster/home/eharve06/resnet50_ssl_prior'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--alpha', type=float, default=0.9, help='1: SGLD; <1: SGHMC')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='directory to save checkpoints (default: None)')
    parser.add_argument('--data_dir', type=str, default='/cluster/tufts/hugheslab/eharve06/HAM10000', help='directory to dataset (default: "/cluster/tufts/hugheslab/eharve06/HAM10000")')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--prior_dir', type=str, default=None, required=True, help='directory to saved priors (default: None)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--temperature', type=float, default=1.0/50000, help='temperature (default: 1/dataset_size)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight_decay (default: 5e-4)')

    args = parser.parse_args()
    # Set torch random seed
    torch.manual_seed(args.seed)
    # Create checkpoints directory
    makedir_if_not_exist(args.checkpoints_dir)

    # Load labels.csv
    df = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'), index_col='lesion_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    # Subsample data and create folds
    df = df.sample(n=1000, random_state=args.seed)
    df['Fold'] = create_folds(df, index_name='lesion_id', random_state=args.seed)
    # Split folds
    train_df, val_df, test_df = split_folds(df)
    # Create datasets
    train_dataset = ImageDataset2D(train_df)
    val_dataset = ImageDataset2D(val_df, train_dataset.mean_and_std)
    test_dataset = ImageDataset2D(test_df, train_dataset.mean_and_std)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Load model
    checkpoint = torch.load('{}/resnet50_ssl_prior_model.pt'.format(args.prior_dir), map_location=torch.device('cpu'))
    model = resnet50() # Define model
    model.fc = torch.nn.Identity() # Get the classification head off
    model.load_state_dict(checkpoint) # Load the pretrained backbone weights
    num_classes = len(np.unique(train_df.label.to_list()))
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_classes, bias=True) # Put the proper classification head back
    model.to(device)
    
    # Load prior parameters
    mean = torch.load('{}/resnet50_ssl_prior_mean.pt'.format(args.prior_dir))
    variance = torch.load('{}/resnet50_ssl_prior_variance.pt'.format(args.prior_dir))
    cov_factor = torch.load('{}/resnet50_ssl_prior_covmat.pt'.format(args.prior_dir))
    prior_scale = 1e10 # Default from "pretrain your loss"
    prior_eps = 1e-1 # Default from "pretrain your loss"
    variance = prior_scale * variance + prior_eps # Scale the variance

    number_of_samples_prior = 5 # Default from "pretrain your loss"
    cov_mat_sqrt = prior_scale * (cov_factor[:number_of_samples_prior]) # Scale the low rank covariance
    prior_params = {'mean': mean.cpu(), 'variance': variance.cpu(), 'cov_mat_sqr': cov_mat_sqrt.cpu()}

    datasize = len(train_loader.dataset)
    num_batch = datasize/args.batch_size+1
    T = args.epochs*num_batch # Total number of iterations
    lr_scheduler = CosineAnnealingLR(num_batch, T, M=4, lr_0=0.5)
    criterion = GaussianPriorCELossShifted(prior_params)

    columns = ['epoch', 'train_loss', 'train_auroc', 'train_bma_auroc', 
               'val_loss', 'val_auroc', 'val_bma_auroc', 'test_loss', 
               'test_auroc', 'test_bma_auroc', 'lrs']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(args.epochs):
        
        lrs = train_one_epoch(model, prior_params, device, criterion, lr_scheduler, train_loader, epoch, args)
        
        train_loss, train_targets, train_outputs = evaluate(model, prior_params, device, criterion, train_loader)
        val_loss, val_targets, val_outputs = evaluate(model, prior_params, device, criterion, val_loader)
        test_loss, test_targets, test_outputs = evaluate(model, prior_params, device, criterion, test_loader)
        
        # Calculate AUROCs
        train_auroc = get_auroc(to_categorical(train_targets, num_classes), train_outputs)
        val_auroc = get_auroc(to_categorical(val_targets, num_classes), val_outputs)
        test_auroc = get_auroc(to_categorical(test_targets, num_classes), test_outputs)
        
        if (epoch%50)+1>45:
            # Bayesian model average
            train_bma_outputs = bayesian_model_average(model, prior_params, device, criterion, train_loader, args.checkpoints_dir)
            val_bma_outputs = bayesian_model_average(model, prior_params, device, criterion, val_loader, args.checkpoints_dir)
            test_bma_outputs = bayesian_model_average(model, prior_params, device, criterion, test_loader, args.checkpoints_dir)
            
            # Calculate Bayesian model average AUROCs
            train_bma_auroc = get_auroc(to_categorical(train_targets, num_classes), train_bma_outputs)
            val_bma_auroc = get_auroc(to_categorical(val_targets, num_classes), val_bma_outputs)
            test_bma_auroc = get_auroc(to_categorical(test_targets, num_classes), test_bma_outputs)
            
            # Save 5 models per cycle
            if np.mean(model_history_df.loc[epoch-1].val_bma_auroc) < np.mean(val_bma_auroc):
                print('Saving model_epoch={}.pt'.format(epoch))
                model.cpu()
                torch.save(model.state_dict(), '{}/model_epoch={}.pt'.format(args.checkpoints_dir, epoch))
                model.to(device)
        else:
            train_bma_auroc = [0.0]*num_classes if epoch == 0 else model_history_df.loc[epoch-1].train_bma_auroc
            val_bma_auroc = [0.0]*num_classes if epoch == 0 else model_history_df.loc[epoch-1].val_bma_auroc
            test_bma_auroc = [0.0]*num_classes if epoch == 0 else model_history_df.loc[epoch-1].test_bma_auroc
            
        # Append evaluation metrics to DataFrame
        row = [epoch+1, train_loss, train_auroc, train_bma_auroc, val_loss, 
               val_auroc, val_bma_auroc, test_loss, test_auroc, test_bma_auroc, 
               lrs]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])

        model_history_df.to_csv('{}/model_history.csv'.format(args.checkpoints_dir))