import os
import itertools
import numpy as np
import pandas as pd
# PyTorch
import torch
import torchvision
import matplotlib.pyplot as plt

import sys
sys.path.append('../src/')
# Importing our custom module(s)
import utils
import losses

def get_df(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    return df

def get_last_epoch(df):
    return df.iloc[-1]
    
def get_best_hyperparameters(experiments_path, lr_0s, ns, prior_scales, prior_type, random_states, weight_decays):
    columns = ['lr_0', 'n', 'prior_scale', 'prior_type', 'random_state', 'val_acc', 'weight_decay']
    df = pd.DataFrame(columns=columns)
    for n, random_state in itertools.product(ns, random_states):
        best_val_nll = np.inf
        best_hyperparameters = None
        for lr_0, prior_scale, weight_decay in itertools.product(lr_0s, prior_scales, weight_decays):
            if prior_scale:
                model_name = '{}_lr_0={}_n={}_prior_scale={}_random_state={}_weight_decay={}'\
                .format(prior_type, lr_0, n, prior_scale, random_state, weight_decay)
            else:
                model_name = '{}_lr_0={}_n={}_random_state={}_weight_decay={}'\
                .format(prior_type, lr_0, n, random_state, weight_decay)
            path =  '{}/{}.csv'.format(experiments_path, model_name)
            val_nll = get_val_nll(get_df(path))
            val_acc = get_val_acc(get_df(path))
            if val_nll < best_val_nll: best_val_nll = val_nll; best_hyperparameters = [lr_0, n, prior_scale, prior_type, random_state, val_acc, weight_decay]
        df.loc[df.shape[0]] = best_hyperparameters
    return df

def get_val_acc(df):
    return df.val_or_test_acc.values[-1]

def get_val_nll(df):
    return df.val_or_test_nll.values[-1]

def interpolate_checkpoints(first_checkpoint, second_checkpoint, n=41):
    interpolations = [{} for _ in range(n)]
    alphas = np.linspace(1.5, -0.5, num=n)
    betas =  np.linspace(-0.5, 1.5, num=n)
    for interpolation_index, (alpha, beta) in enumerate(zip(alphas, betas)):
        for key in first_checkpoint.keys():
            interpolations[interpolation_index][key] = (alpha * first_checkpoint[key].detach().clone()) + (beta * second_checkpoint[key].detach().clone()).detach().clone()
            if 'running_var' in key and alpha > 1.0:
                interpolations[interpolation_index][key] = first_checkpoint[key].detach().clone()
            if 'running_var' in key and alpha < 0.0:
                interpolations[interpolation_index][key] = second_checkpoint[key].detach().clone()
    return interpolations

if __name__=='__main__':
    # Nonlearned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10_Copy1'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    prior_scales = [None]
    prior_type = 'nonlearned'
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    nonlearned_hyperparameters = get_best_hyperparameters(experiments_path, lr_0s, ns, prior_scales, prior_type, random_states, weight_decays)
    
    for (nonlearned_index, nonlearned_row) in nonlearned_hyperparameters.iterrows():
        num_heads = 10
        random_state = int(nonlearned_row.random_state)
        # Finetuned model
        experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1'
        model_name = 'nonlearned_lr_0={}_n={}_random_state={}_weight_decay={}'\
        .format(nonlearned_row.lr_0, int(nonlearned_row.n), int(nonlearned_row.random_state), nonlearned_row.weight_decay)
        finetuned_checkpoint = torch.load('{}/{}.pth'.format(experiments_path, model_name), map_location=torch.device('cpu'))
        # Pretrained checkpoint
        prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'
        pretrained_checkpoint = torch.load('{}/resnet50_ssl_prior_model.pt'.format(prior_path), map_location=torch.device('cpu'))
        pretrained_checkpoint['fc.weight'] = finetuned_checkpoint['fc.weight']
        pretrained_checkpoint['fc.bias'] = finetuned_checkpoint['fc.bias']
        interpolations = interpolate_checkpoints(pretrained_checkpoint, finetuned_checkpoint)
        
        dataset_path = '/cluster/tufts/hugheslab/eharve06/CIFAR-10'
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=1000, tune=False, random_state=random_state)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True)
        model.to(device)

        ce = torch.nn.CrossEntropyLoss()
        criterion = losses.CustomCELoss(ce)

        train_losses, val_or_test_losses = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_loss, train_nll, train_prior, train_acc = utils.evaluate(model, criterion, train_loader)
            train_losses.append(train_loss)
            print(train_loss)
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = utils.evaluate(model, criterion, val_or_test_loader)
            val_or_test_losses.append(val_or_test_nll)
            print(val_or_test_nll)
            print()
            
        train_losses = torch.tensor(train_losses)
        val_or_test_losses = torch.tensor(val_or_test_losses)
        torch.save(train_losses, './nonlearned_train_basin_random_state={}.pth'.format(random_state))
        torch.save(val_or_test_losses, './nonlearned_test_basin_random_state={}.pth'.format(random_state))

    # Learned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10_Copy1'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    prior_scales = np.logspace(0, 9, num=10)
    prior_type = 'learned'
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    learned_hyperparameters = get_best_hyperparameters(experiments_path, lr_0s, ns, prior_scales, prior_type, random_states, weight_decays)
    
    for (learned_index, learned_row) in learned_hyperparameters.iterrows():
        num_heads = 10
        random_state = int(learned_row.random_state)
        # Finetuned model
        experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10_Copy1'
        model_name = 'learned_lr_0={}_n={}_prior_scale={}_random_state={}_weight_decay={}'\
        .format(learned_row.lr_0, int(learned_row.n), learned_row.prior_scale, int(learned_row.random_state), learned_row.weight_decay)
        finetuned_checkpoint = torch.load('{}/{}.pth'.format(experiments_path, model_name), map_location=torch.device('cpu'))
        # Pretrained checkpoint
        prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'
        pretrained_checkpoint = torch.load('{}/resnet50_ssl_prior_model.pt'.format(prior_path), map_location=torch.device('cpu'))
        pretrained_checkpoint['fc.weight'] = finetuned_checkpoint['fc.weight']
        pretrained_checkpoint['fc.bias'] = finetuned_checkpoint['fc.bias']
        interpolations = interpolate_checkpoints(pretrained_checkpoint, finetuned_checkpoint)
        
        dataset_path = '/cluster/tufts/hugheslab/eharve06/CIFAR-10'
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=1000, tune=False, random_state=random_state)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True)
        model.to(device)

        prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'
        number_of_samples_prior = 5 # Default        
        loc = torch.load('{}/resnet50_ssl_prior_mean.pt'.format(prior_path))
        cov_factor = torch.load('{}/resnet50_ssl_prior_covmat.pt'.format(prior_path))
        cov_factor = learned_row.prior_scale * (cov_factor[:number_of_samples_prior]) # Scale the low rank covariance
        cov_diag = torch.load('{}/resnet50_ssl_prior_variance.pt'.format(prior_path))
        prior_eps = 1e-1 # Default from "Pre-Train Your Loss"
        cov_diag = learned_row.prior_scale * cov_diag + prior_eps # Scale the variance
        ce = torch.nn.CrossEntropyLoss()
        criterion = losses.GaussianPriorCELossShifted(ce, loc.cpu(), cov_factor.t().cpu(), cov_diag.cpu())

        train_losses, val_or_test_losses = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_loss, train_nll, train_prior, train_acc = utils.evaluate(model, criterion, train_loader)
            train_losses.append(train_loss)
            print(train_loss)
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = utils.evaluate(model, criterion, val_or_test_loader)
            val_or_test_losses.append(val_or_test_nll)
            print(val_or_test_nll)
            print()
            
        train_losses = torch.tensor(train_losses)
        val_or_test_losses = torch.tensor(val_or_test_losses)
        torch.save(train_losses, './learned_train_basin_random_state={}.pth'.format(random_state))
        torch.save(val_or_test_losses, './learned_test_basin_random_state={}.pth'.format(random_state))        