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

def get_val_nll(df):
    return df.val_or_test_nll.values[-1]

def get_val_acc(df):
    return df.val_or_test_acc.values[-1]

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
                model_name = f'{prior_type}_lr_0={lr_0}_n={n}_prior_scale={prior_scale}_random_state={random_state}_weight_decay={weight_decay}'
            else:
                model_name = f'{prior_type}_lr_0={lr_0}_n={n}_random_state={random_state}_weight_decay={weight_decay}'
            path =  f'{experiments_path}/{model_name}.csv'
            val_nll = get_val_nll(get_df(path))
            val_acc = get_val_acc(get_df(path))
            if val_nll < best_val_nll: best_val_nll = val_nll; best_hyperparameters = [lr_0, n, prior_scale, prior_type, random_state, val_acc, weight_decay]
        df.loc[df.shape[0]] = best_hyperparameters
    return df

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
    # Learned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    prior_scales = np.logspace(0, 4, num=5)
    prior_type = 'learned'
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    learned_hyperparameters = get_best_hyperparameters(experiments_path, lr_0s, ns, prior_scales, prior_type, random_states, weight_decays)
    # Nonlearned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    prior_scales = [None]
    prior_type = 'nonlearned'
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    nonlearned_hyperparameters = get_best_hyperparameters(experiments_path, lr_0s, ns, prior_scales, prior_type, random_states, weight_decays)
    
    for (learned_index, learned_row), (nonlearned_index, nonlearned_row) in zip(learned_hyperparameters.iterrows(), nonlearned_hyperparameters.iterrows()):
        # Load learned checkpoint
        experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10'
        model_name = f'learned_lr_0={learned_row.lr_0}_n={learned_row.n}_prior_scale={learned_row.prior_scale}_random_state={learned_row.random_state}_weight_decay={learned_row.weight_decay}'
        learned_checkpoint = torch.load(f'{experiments_path}/{model_name}.pth', map_location=torch.device('cpu'))
        # Load nonlearned checkpoint
        experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10'
        model_name = f'nonlearned_lr_0={nonlearned_row.lr_0}_n={nonlearned_row.n}_random_state={nonlearned_row.random_state}_weight_decay={nonlearned_row.weight_decay}'
        nonlearned_checkpoint = torch.load(f'{experiments_path}/{model_name}.pth', map_location=torch.device('cpu'))

        interpolations = interpolate_checkpoints(nonlearned_checkpoint, learned_checkpoint)
        assert int(learned_row.random_state) == int(nonlearned_row.random_state), 'Expected random_state in each row to be the same'
        random_state = int(learned_row.random_state)
        dataset_path = '/cluster/tufts/hugheslab/eharve06/CIFAR-10'
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=1000, tune=False, random_state=random_state)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
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
        torch.save(train_losses, './nonlearned_train_interpolation_random_state={}.pth'.format(random_state))
        torch.save(val_or_test_losses, './nonlearned_test_interpolation_random_state={}.pth'.format(random_state))
        
        prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'
        number_of_samples_prior = 5 # Default        
        loc = torch.load(f'{prior_path}/resnet50_ssl_prior_mean.pt')
        cov_factor = torch.load(f'{prior_path}/resnet50_ssl_prior_covmat.pt')
        cov_factor = learned_row.prior_scale * (cov_factor[:number_of_samples_prior]) # Scale the low rank covariance
        cov_diag = torch.load(f'{prior_path}/resnet50_ssl_prior_variance.pt')
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
        torch.save(train_losses, f'./learned_train_interpolation_random_state={random_state}.pth')
        torch.save(val_or_test_losses, f'./learned_test_interpolation_random_state={random_state}.pth')