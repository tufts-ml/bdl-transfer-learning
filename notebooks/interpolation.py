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
import CIFAR10_utils as utils
import losses

def get_df(path):
    df = pd.read_csv(path, index_col='Unnamed: 0')
    return df

def get_last_epoch(df):
    return df.iloc[-1]
    
def get_learned_hyperparameters(experiments_path, lr_0s, ns, prior_scales, random_states, weight_decays):
    columns = ['lr_0', 'n', 'prior_scale', 'random_state', 'weight_decay']
    df = pd.DataFrame(columns=columns)
    for n, random_state in itertools.product(ns, random_states):
        best_val_nll = np.inf
        best_hyperparameters = None
        for prior_scale in prior_scales:
            best_val_loss = np.inf
            best_model_name = None
            best_lr_0 = None
            best_weight_decay = None
            for lr_0, weight_decay in itertools.product(lr_0s, weight_decays):
                model_name = 'learned_lr_0={}_n={}_prior_scale={}_random_state={}_weight_decay={}'\
                .format(lr_0, n, prior_scale, random_state, weight_decay)
                path =  '{}/{}.csv'.format(experiments_path, model_name)
                val_loss = get_val_loss(get_df(path))
                if val_loss < best_val_loss: best_val_loss = val_loss; best_model_name = model_name; best_lr_0 = lr_0; best_weight_decay = weight_decay
            path = '{}/{}.csv'.format(experiments_path, best_model_name)
            val_nll = get_val_nll(get_df(path))
            if val_nll < best_val_nll: best_val_nll = val_nll; best_hyperparameters = [best_lr_0, n, prior_scale, random_state, best_weight_decay]
        df.loc[df.shape[0]] = best_hyperparameters
    return df

def get_nonlearned_hyperparameters(experiments_path, lr_0s, ns, random_states, weight_decays):
    columns = ['lr_0', 'n', 'random_state', 'weight_decay']
    df = pd.DataFrame(columns=columns)
    for n, random_state in itertools.product(ns, random_states):
        best_val_nll = np.inf
        best_hyperparameters = None
        for lr_0, weight_decay in itertools.product(lr_0s, weight_decays):
            model_name = 'nonlearned_lr_0={}_n={}_random_state={}_weight_decay={}'\
            .format(lr_0, n, random_state, weight_decay)
            path =  '{}/{}.csv'.format(experiments_path, model_name)
            val_nll = get_val_nll(get_df(path))
            if val_nll < best_val_nll: best_val_nll = val_nll; best_hyperparameters = [lr_0, n, random_state, weight_decay]
        df.loc[df.shape[0]] = best_hyperparameters
    return df

def get_val_loss(df):
    return df.val_or_test_loss.values[-1]

def get_val_nll(df):
    return df.val_or_test_nll.values[-1]

def interpolate_checkpoints(first_checkpoint, second_checkpoint, n=21):
    interpolations = [{} for _ in range(n)]
    alphas = np.linspace(1, 0, num=n)
    betas =  np.linspace(0, 1, num=n)
    for interpolation_index, (alpha, beta) in enumerate(zip(alphas, betas)):
        for key in first_checkpoint.keys():
            interpolations[interpolation_index][key] = (alpha * first_checkpoint[key].detach().clone()) + (beta * second_checkpoint[key].detach().clone()).detach().clone()
    return interpolations

if __name__=='__main__':
    # Learned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    prior_scales = np.logspace(0, 9, num=10)
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    learned_hyperparameters = get_learned_hyperparameters(experiments_path, lr_0s, ns, prior_scales, random_states, weight_decays)
    # Nonlearned
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/tuned_CIFAR-10'
    lr_0s = np.logspace(-1, -4, num=4)
    ns = [1000]
    random_states = [1001, 2001, 3001]
    weight_decays = np.append(np.logspace(-2, -6, num=5), 0)
    nonlearned_hyperparameters = get_nonlearned_hyperparameters(experiments_path, lr_0s, ns, random_states, weight_decays)
    
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10'
    for (learned_index, learned_row), (nonlearned_index, nonlearned_row) in zip(learned_hyperparameters.iterrows(), nonlearned_hyperparameters.iterrows()):
        # Load learned checkpoint
        model_name = 'learned_lr_0={}_n={}_prior_scale={}_random_state={}_weight_decay={}'\
        .format(learned_row.lr_0, int(learned_row.n), learned_row.prior_scale, int(learned_row.random_state), learned_row.weight_decay)
        learned_checkpoint = torch.load('{}/{}.pth'.format(experiments_path, model_name), map_location=torch.device('cpu'))
        # Load nonlearned checkpoint
        model_name = 'nonlearned_lr_0={}_n={}_random_state={}_weight_decay={}'\
        .format(nonlearned_row.lr_0, int(nonlearned_row.n), int(nonlearned_row.random_state), nonlearned_row.weight_decay)
        nonlearned_checkpoint = torch.load('{}/{}.pth'.format(experiments_path, model_name), map_location=torch.device('cpu'))
        
        # Interpolate checkpoints
        #learned_checkpoint = {}
        #for key in nonlearned_checkpoint.keys():
        #    learned_checkpoint[key] = torch.zeros(nonlearned_checkpoint[key].shape)

        interpolations = interpolate_checkpoints(nonlearned_checkpoint, learned_checkpoint)
        assert int(learned_row.random_state) == int(nonlearned_row.random_state), 'Expected random_state in each row to be the same'
        random_state = int(learned_row.random_state)
        dataset_path = '/cluster/tufts/hugheslab/eharve06/CIFAR-10'
        train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=1000, tune=False, random_state=random_state, use_train_transform=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)
        model.to(device)

        prior_params = {'mean': torch.Tensor([0]), 'variance': torch.Tensor([0]), 'cov_mat_sqr': torch.Tensor([0])}
        ce = torch.nn.CrossEntropyLoss()
        criterion = losses.CustomCELoss(ce)

        train_losses, val_or_test_losses = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_loss, train_nll, train_prior, train_acc = utils.evaluate(model, prior_params, criterion, train_loader)
            train_losses.append(train_loss)
            print(train_loss)
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = utils.evaluate(model, prior_params, criterion, val_or_test_loader)
            val_or_test_losses.append(val_or_test_loss)
            print(val_or_test_loss)
            print()
            
        train_losses = torch.tensor(train_losses)
        val_or_test_losses = torch.tensor(val_or_test_losses)
        torch.save(train_losses, './nonlearned_train_interpolation_random_state={}.pth'.format(random_state))
        torch.save(val_or_test_losses, './nonlearned_test_interpolation_random_state={}.pth'.format(random_state))
        
        prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'
        number_of_samples_prior = 5 # Default        
        mean = torch.load('{}/resnet50_ssl_prior_mean.pt'.format(prior_path))
        variance = torch.load('{}/resnet50_ssl_prior_variance.pt'.format(prior_path))
        cov_factor = torch.load('{}/resnet50_ssl_prior_covmat.pt'.format(prior_path))
        prior_eps = 1e-1 # Default from "Pre-Train Your Loss"
        variance = learned_row.prior_scale * variance + prior_eps # Scale the variance
        cov_mat_sqrt = learned_row.prior_scale * (cov_factor[:number_of_samples_prior]) # Scale the low rank covariance
        prior_params = {'mean': mean.cpu(), 'variance': variance.cpu(), 'cov_mat_sqr': cov_mat_sqrt.cpu()}
        ce = torch.nn.CrossEntropyLoss()
        criterion = losses.GaussianPriorCELossShifted(ce, prior_params)
        
        train_losses, val_or_test_losses = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_loss, train_nll, train_prior, train_acc = utils.evaluate(model, prior_params, criterion, train_loader)
            train_losses.append(train_loss)
            print(train_loss)
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = utils.evaluate(model, prior_params, criterion, val_or_test_loader)
            val_or_test_losses.append(val_or_test_loss)
            print(val_or_test_loss)
            print()
            
        train_losses = torch.tensor(train_losses)
        val_or_test_losses = torch.tensor(val_or_test_losses)
        torch.save(train_losses, './learned_train_interpolation_random_state={}.pth'.format(random_state))
        torch.save(val_or_test_losses, './learned_test_interpolation_random_state={}.pth'.format(random_state))