import re
import copy
import numpy as np
# PyTorch
import torch
import torchvision

import sys
sys.path.append('../src/')
# Importing our custom module(s)
import losses
import utils

def interpolate_checkpoints(first_checkpoint, second_checkpoint, n=41):
    interpolations = [{} for _ in range(n)]
    alphas = np.linspace(1.5, -0.5, num=n)
    for index, alpha in enumerate(alphas):
        for key in first_checkpoint.keys():
            interpolations[index][key] = (alpha * copy.deepcopy(first_checkpoint[key].detach())) + ((1-alpha) * copy.deepcopy(second_checkpoint[key].detach()))
            if 'running_var' in key and alpha > 1.0:
                interpolations[index][key] = copy.deepcopy(first_checkpoint[key].detach())
            elif 'running_var' in key and alpha < 0.0:
                interpolations[index][key] = copy.deepcopy(second_checkpoint[key].detach())
    return interpolations

if __name__=='__main__':
    experiments_path = '/cluster/tufts/hugheslab/eharve06/bdl-transfer-learning/experiments/retrained_CIFAR-10'
    dataset_path = '/cluster/tufts/hugheslab/eharve06/CIFAR-10'
    prior_path = '/cluster/tufts/hugheslab/eharve06/resnet50_ssl_prior'

    # Best models at n=1000 (see CIFAR-10.ipynb)
    std_prior_model_names = [
        'StdPrior_lr_0=0.0001_n=1000_random_state=1001_weight_decay=0.0001',
        'StdPrior_lr_0=0.01_n=1000_random_state=2001_weight_decay=1e-05',
        'StdPrior_lr_0=0.01_n=1000_random_state=3001_weight_decay=0.001',
    ]
    learned_prior_iso_model_names = [
        'adapted_lr_0=0.1_n=1000_random_state=1001_weight_decay=0.01', 
        'adapted_lr_0=0.1_n=1000_random_state=2001_weight_decay=0.01', 
        'adapted_lr_0=0.01_n=1000_random_state=3001_weight_decay=1e-05'
    ]
    learned_prior_lr_model_names = [
        'LearnedPriorLR_bb_weight_decay=100.0_clf_weight_decay=0.0_lr_0=0.01_n=1000_random_state=1001', 
        'LearnedPriorLR_bb_weight_decay=100.0_clf_weight_decay=0.001_lr_0=0.1_n=1000_random_state=2001',
        'LearnedPriorLR_bb_weight_decay=10.0_clf_weight_decay=0.001_lr_0=0.1_n=1000_random_state=3001',
    ]
    best_model_names = [
        'LearnedPriorLR_bb_weight_decay=10.0_clf_weight_decay=0.01_lr_0=0.01_n=50000_random_state=1001', 
        'LearnedPriorLR_bb_weight_decay=10.0_clf_weight_decay=0.01_lr_0=0.01_n=50000_random_state=2001', 
        'LearnedPriorLR_bb_weight_decay=10.0_clf_weight_decay=0.01_lr_0=0.01_n=50000_random_state=3001'
    ]

    for std_prior_model_name, best_model_name in zip(std_prior_model_names, best_model_names):
        std_prior_checkpoint = torch.load(f'{experiments_path}/{std_prior_model_name}.pt', map_location=torch.device('cpu'))
        best_checkpoint = torch.load(f'{experiments_path}/{best_model_name}.pt', map_location=torch.device('cpu'))
        interpolations = interpolate_checkpoints(std_prior_checkpoint, best_checkpoint)

        # StdPrior
        pattern = re.compile(r'(\w+)_lr_0=([\d.]+)_n=(\d+)_random_state=(\d+)_weight_decay=([\d.]+(?:e[-+]?\d+)?)')
        match = pattern.match(std_prior_model_name)
        prior_type, lr_0, n, random_state, weight_decay = match.groups()
        lr_0, n, random_state, weight_decay = float(lr_0), int(n), int(random_state), float(weight_decay)

        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=n, tune=False, random_state=random_state)
        # PyTorch DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)

        num_heads = 10
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(f'{prior_path}/resnet50_ssl_prior_model.pt', map_location=torch.device('cpu'))
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True)
        model.to(device)

        # prior_type == 'StdPrior'
        criterion = losses.L2NormLoss(weight_decay=weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_0, momentum=0.9, weight_decay=0, nesterov=True)

        train_losses, val_or_test_nlls = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_metrics = utils.evaluate(model, criterion, train_loader)
            train_losses.append(train_metrics['loss'])
            print(train_metrics['loss'])
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader)
            val_or_test_nlls.append(val_or_test_metrics['nll'])
            print(val_or_test_metrics['nll'])
            print()

        train_losses = torch.tensor(train_losses)
        val_or_test_nlls = torch.tensor(val_or_test_nlls)
        torch.save(train_losses, f'./nonlearned_train_new_interpolation_random_state={random_state}.pt')
        torch.save(val_or_test_nlls, f'./nonlearned_test_new_interpolation_random_state={random_state}.pt')

    for learned_prior_iso_model_name, best_model_name in zip(learned_prior_iso_model_names, best_model_names):
        learned_prior_iso_checkpoint = torch.load(f'{experiments_path}/{learned_prior_iso_model_name}.pt', map_location=torch.device('cpu'))
        best_checkpoint = torch.load(f'{experiments_path}/{best_model_name}.pt', map_location=torch.device('cpu'))
        interpolations = interpolate_checkpoints(learned_prior_iso_checkpoint, best_checkpoint)

        # LearnedPriorIso
        pattern = re.compile(r'(\w+)_lr_0=([\d.]+)_n=(\d+)_random_state=(\d+)_weight_decay=([\d.]+(?:e[-+]?\d+)?)')
        match = pattern.match(learned_prior_iso_model_name)
        prior_type, lr_0, n, random_state, weight_decay = match.groups()
        lr_0, n, random_state, weight_decay = float(lr_0), int(n), int(random_state), float(weight_decay)
        
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=n, tune=False, random_state=random_state)
        # PyTorch DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)

        num_heads = 10
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(f'{prior_path}/resnet50_ssl_prior_model.pt', map_location=torch.device('cpu'))
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True)
        model.to(device)

        # prior_type == 'LearnedPriorIso'
        loc = torch.load(f'{prior_path}/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu'))
        loc = torch.cat((loc, torch.zeros((2048*num_heads)+num_heads)))
        criterion = losses.MAPAdaptationLoss(loc, weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_0, momentum=0.9, weight_decay=0, nesterov=True)

        train_losses, val_or_test_nlls = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_metrics = utils.evaluate(model, criterion, train_loader)
            train_losses.append(train_metrics['loss'])
            print(train_metrics['loss'])
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader)
            val_or_test_nlls.append(val_or_test_metrics['nll'])
            print(val_or_test_metrics['nll'])
            print()

        train_losses = torch.tensor(train_losses)
        val_or_test_nlls = torch.tensor(val_or_test_nlls)
        torch.save(train_losses, f'./adapted_train_new_interpolation_random_state={random_state}.pt')
        torch.save(val_or_test_nlls, f'./adapted_test_new_interpolation_random_state={random_state}.pt')

    for learned_prior_lr_model_name, best_model_name in zip(learned_prior_lr_model_names, best_model_names):
        learned_prior_lr_checkpoint = torch.load(f'{experiments_path}/{learned_prior_lr_model_name}.pt', map_location=torch.device('cpu'))
        best_checkpoint = torch.load(f'{experiments_path}/{best_model_name}.pt', map_location=torch.device('cpu'))
        interpolations = interpolate_checkpoints(learned_prior_lr_checkpoint, best_checkpoint)

        # LearnedPriorLR
        pattern = re.compile(r'(\w+)_bb_weight_decay=([\d.]+)_clf_weight_decay=([\d.]+(?:e[-+]?\d+)?)_lr_0=([\d.]+)_n=(\d+)_random_state=(\d+)')
        match = pattern.match(learned_prior_lr_model_name)
        prior_type, bb_weight_decay, clf_weight_decay, lr_0, n, random_state = match.groups()
        bb_weight_decay, clf_weight_decay, lr_0, n, random_state = float(bb_weight_decay), float(clf_weight_decay), float(lr_0), int(n), int(random_state)
        
        augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_cifar10_datasets(root=dataset_path, n=n, tune=False, random_state=random_state)
        # PyTorch DataLoaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
        val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=128)

        num_heads = 10
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(f'{prior_path}/resnet50_ssl_prior_model.pt', map_location=torch.device('cpu'))
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Identity()
        model.load_state_dict(checkpoint)
        model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True)
        model.to(device)

        # All LearnedPriorLR experiments were run with k = 5 and prior_eps = 0.1
        k, prior_eps = 5, 0.1

        # prior_type == 'LearnedPriorLR'
        bb_prior = {
            'cov_diag': torch.load(f'{prior_path}/resnet50_ssl_prior_variance.pt', map_location=torch.device('cpu')),
            'cov_factor': torch.load(f'{prior_path}/resnet50_ssl_prior_covmat.pt', map_location=torch.device('cpu')),
            'prior_eps': prior_eps,
            'loc': torch.load(f'{prior_path}/resnet50_ssl_prior_mean.pt', map_location=torch.device('cpu')),
        }
        bb_prior['cov_factor'] = bb_prior['cov_factor'][:k]
        # $\Sigma = \frac{1}{2} ( \Sigma_{\text{diag}} + \Sigma_{\text{LR}} )$
        bb_prior['cov_diag'] = (1/2)*bb_prior['cov_diag']
        bb_prior['cov_factor'] = np.sqrt(1/2)*bb_prior['cov_factor']
        # $\Sigma_{\text{LR}} = \frac{1}{k-1} Q Q^T$
        bb_prior['cov_factor'] = np.sqrt(1/(k-1))*bb_prior['cov_factor']
        clf_prior = {
            'cov_diag': torch.ones((2048*num_heads)+num_heads),
            'cov_factor': torch.zeros(1, (2048*num_heads)+num_heads),
            'loc': torch.zeros((2048*num_heads)+num_heads),
        }
        clf_weight_decay = 0 if clf_weight_decay == 0 else 1/(len(augmented_train_dataset) * clf_weight_decay)
        criterion = losses.MAPTransferLearning(bb_prior, bb_weight_decay, clf_prior, clf_weight_decay, device, len(augmented_train_dataset))

        train_losses, val_or_test_nlls = [], []
        for checkpoint in interpolations:
            model.load_state_dict(checkpoint)
            train_metrics = utils.evaluate(model, criterion, train_loader)
            train_losses.append(train_metrics['loss'])
            print(train_metrics['loss'])
            val_or_test_metrics = utils.evaluate(model, criterion, val_or_test_loader)
            val_or_test_nlls.append(val_or_test_metrics['nll'])
            print(val_or_test_metrics['nll'])
            print()

        train_losses = torch.tensor(train_losses)
        val_or_test_nlls = torch.tensor(val_or_test_nlls)
        torch.save(train_losses, f'./learned_train_new_interpolation_random_state={random_state}.pt')
        torch.save(val_or_test_nlls, f'./learned_test_new_interpolation_random_state={random_state}.pt')
