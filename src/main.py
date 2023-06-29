'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
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

from losses import GaussianPriorCELossShifted
from utils import *

# python bdl-transfer-learning/src/main.py --checkpoints_dir='/cluster/home/eharve06/bdl-transfer-learning/checkpoints' --prior_dir='/cluster/home/eharve06/resnet50_ssl_prior' --data_dir='/cluster/tufts/hugheslab/eharve06/CIFAR10'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='directory to save checkpoints (default: None)')
    parser.add_argument('--prior_dir', type=str, default=None, required=True, help='directory to saved priors (default: None)')
    parser.add_argument('--data_dir', type=str, default=None, required=True, metavar='PATH',
                        help='directory to save dataset (default: None)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='1: SGLD; <1: SGHMC')
    parser.add_argument('--device_id',type = int, help = 'device id to use')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--temperature', type=float, default=1./50000,
                        help='temperature (default: 1/dataset_size)')

    args = parser.parse_args()
    device_id = args.device_id
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # resize shorter
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

    # Model
    print('==> Building model..')
    print("Working with pretrained prior!")

    path = '{}/resnet50_ssl_prior'.format(args.prior_dir)
    checkpoint = torch.load(path+'_model.pt', map_location=torch.device('cpu'))

    model = resnet50() # Define model
    model.fc = torch.nn.Identity() # Get the classification head off
    model.load_state_dict(checkpoint) # Load the pretrained backbone weights
    model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True) # Put the proper classification head back

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
    mt = 0
    print(datasize)

    lrs = []
    metrics = []
    for epoch in range(args.epochs):
        lrs_row = train(model, prior_params, device, criterion, lr_scheduler, train_loader, epoch)
        lrs.append(lrs_row)
        av_test_loss, acc = test(model, prior_params, device, criterion, test_loader)
        metrics.append([av_test_loss, acc])
        if (epoch%50)+1>47: # save 3 models per cycle
            print('save!')
            model.cpu()
            torch.save(model.state_dict(), args.dir + '/cifar_csghmc_%i.pt'%(mt))
            mt += 1
            model.to(device)

        pd.DataFrame(lrs).to_csv('./learning_rates_clr.csv')
        pd.DataFrame(metrics,columns=['loss','acc']).to_csv('./perf_metrics_clr.csv')