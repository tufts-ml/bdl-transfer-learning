'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import ast

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import math
from torchvision.models import resnet50
import pandas as pd
from torch.autograd import Variable
import numpy as np
import random

from evaluation_metrics import *
from folds import *
from losses import *
from utils import *

# python bdl-transfer-learning/src/main.py --checkpoints_dir='/cluster/home/eharve06/bdl-transfer-learning/checkpoints' --prior_dir='/cluster/home/eharve06/resnet50_ssl_prior'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='directory to save checkpoints (default: None)')
    parser.add_argument('--prior_dir', type=str, default=None, required=True, help='directory to saved priors (default: None)')
    parser.add_argument('--data_dir', type=str, default='/cluster/tufts/hugheslab/eharve06/HAM10000',
                        help='directory to save dataset (default: None)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='1: SGLD; <1: SGHMC')
    parser.add_argument('--device_id',type = int, help = 'device id to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--temperature', type=float, default=1./50000,
                        help='temperature (default: 1/dataset_size)')

    args = parser.parse_args()
    # Set torch random seed
    torch.manual_seed(args.seed)

    # Load labesl.csv
    df = pd.read_csv(os.path.join(args.data_dir, 'labels.csv'), index_col='lesion_id')
    df.label = df.label.apply(lambda string: ast.literal_eval(string))
    # Subsample data and create folds
    df = df.sample(n=1000, random_state=args.seed)
    df['Fold'] = create_folds(df, index_name='lesion_id', random_state=args.seed)
    # Split folds
    train_df, val_df, test_df = split_folds(df)
    # Create datasets
    train_dataset = ImageDataset(train_df)
    val_dataset = ImageDataset(val_df)
    test_dataset = ImageDataset(test_df)
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
    
    # Model
    print('==> Building model..')
    print("Working with pretrained prior!")

    path = '{}/resnet50_ssl_prior'.format(args.prior_dir)
    checkpoint = torch.load(path+'_model.pt', map_location=torch.device('cpu'))

    model = resnet50() # Define model
    model.fc = torch.nn.Identity() # Get the classification head off
    model.load_state_dict(checkpoint) # Load the pretrained backbone weights
    num_labels = np.array(train_df.label.to_list()).shape[-1]
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_labels, bias=True) # Put the proper classification head back

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model.to(device)
    if device.type == 'cuda':
        cudnn.benchmark = True
        cudnn.deterministic = True

    #### Load prior parameters
    print("Loading prior parameters")
    mean = torch.load(path + '_mean.pt')
    variance = torch.load(path + '_variance.pt')
    cov_factor = torch.load(path + '_covmat.pt')
    print("Loaded")
    print("Parameter space dimension:", mean.shape)
    prior_scale = 1e10 # default from "pretrain your loss"
    prior_eps = 1e-1 # default from "pretrain your loss"
    ### scale the variance
    variance = prior_scale * variance + prior_eps

    number_of_samples_prior = 5 # default from "pretrain your loss"
    ### scale the low rank covariance
    cov_mat_sqrt = prior_scale * (cov_factor[:number_of_samples_prior])
    prior_params = {'mean': mean.cpu(), 'variance': variance.cpu(), 'cov_mat_sqr': cov_mat_sqrt.cpu()}

    weight_decay = 5e-4
    datasize = len(train_loader.dataset)
    num_batch = datasize/args.batch_size+1
    T = args.epochs*num_batch # total number of iterations
    M = 4 # number of cycles
    lr_0 = 0.5 # initial lr
    lr_scheduler = CosineAnnealingLR(num_batch, T, M, lr_0)
    criterion = GaussianPriorCELossShifted(prior_params)

    columns = ['epoch', 'train_loss', 'train_BA', 'train_auroc', 'val_loss', 
               'val_BA', 'val_auroc', 'test_loss', 'test_BA', 'test_auroc', 'lrs']
    columns = ['epoch', 'train_loss', 'train_auroc', 'val_loss', 'val_auroc', 
               'test_loss', 'test_auroc', 'lrs']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(args.epochs):
        lrs = train_one_epoch(model, prior_params, device, criterion, lr_scheduler, train_loader, epoch)
        
        train_loss, train_targets, train_outputs = evaluate(model, prior_params, device, criterion, train_loader)
        val_loss, val_targets, val_outputs = evaluate(model, prior_params, device, criterion, val_loader)
        test_loss, test_targets, test_outputs = evaluate(model, prior_params, device, criterion, test_loader)
        
        # Calculate AUROCs
        train_auroc = get_auroc(train_targets, train_outputs)
        val_auroc = get_auroc(val_targets, val_outputs)
        test_auroc = get_auroc(test_targets, test_outputs)
        
        # Append evaluation metrics to DataFrame
        row = [epoch+1, train_loss, train_auroc, val_loss, val_auroc, test_loss, test_auroc, lrs]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        if (epoch%50)+1>45: # save 5 models per cycle
            model.cpu()
            torch.save(model.state_dict(), '{}/model_epoch={}.pt'.format(args.checkpoints_dir, epoch))
            model.to(device)

        model_history_df.to_csv('{}/model_history.csv'.format(args.checkpoints_dir))