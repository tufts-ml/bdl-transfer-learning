import argparse
import os
import ast
import numpy as np
import pandas as pd
# PyTorch
import torch
import torchvision
import wandb
# Importing our custom module(s)
import folds
import losses
import utils

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='HAM10000_main.py')    
    parser.add_argument('--batch_size', default=128, help='Batch size (default: 128)', type=int)
    parser.add_argument('--dataset_path', help='Path to dataset', required=True, type=str)
    parser.add_argument('--experiments_path', help='Path to save experiments', required=True, type=str)
    parser.add_argument('--lr_0', default=0.5, help='Initial learning rate (default: 0.5)', type=float)
    parser.add_argument('--model_name', help='Model name', required=True, type=str)
    parser.add_argument('--n', default=1000, help='Number of training samples (default: 1000)', type=int)
    parser.add_argument('--number_of_samples_prior', default=5, help='Number of low-rank covariance terms of the prior (default: 5)', type=float)
    parser.add_argument('--prior_eps', default=1e-1, help='Added to prior variance (default: 1e-1)', type=float) # Default from "Pre-Train Your Loss"
    parser.add_argument('--prior_path', help='Path to saved priors', required=True, type=str)
    parser.add_argument('--prior_type', help='Determines criterion', required=True, type=str)
    parser.add_argument('--prior_scale', default=1e10, help='Covariance scaling factor (default: 1e10)', type=float)
    parser.add_argument('--random_state', default=42, help='Random state (default: 42)', type=int)
    parser.add_argument('--save', action='store_true', default=False, help='Whether or not to save the model (default: False)')
    parser.add_argument('--tune', action='store_true', default=False, help='Whether validation or test set is used (default: False)')
    parser.add_argument('--wandb', action='store_true', default=False, help='Whether or not to log to wandb')
    parser.add_argument('--wandb_project', default='test', help='Wandb project name (default: \'test\')', type=str)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')

    args = parser.parse_args()
    print(args)
    # Set torch random seed
    torch.manual_seed(args.random_state)
    # Create checkpoints directory
    utils.makedir_if_not_exist(args.experiments_path)
    # Create sampled HAM10000 datasets
    augmented_train_dataset, train_dataset, val_or_test_dataset = utils.get_ham10000_datasets(root=args.dataset_path, n=args.n, tune=args.tune, random_state=args.random_state)
    # Create dataloaders
    train_loader_shuffled = torch.utils.data.DataLoader(augmented_train_dataset, batch_size=min(args.batch_size, len(augmented_train_dataset)), shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_or_test_loader = torch.utils.data.DataLoader(val_or_test_dataset, batch_size=args.batch_size)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if args.wandb:
        wandb.login()
        os.environ['WANDB_API_KEY'] = '4bfaad8bea054341b5e8729c940150221bdfbb6c'
        wandb.init(
            project = args.wandb_project,
            name = args.model_name,
            config={
                'batch_size': args.batch_size,
                'dataset_path': args.dataset_path,
                'device': device,
                'experiments_path': args.experiments_path,
                'lr_0': args.lr_0,
                'model_name': args.model_name,
                'n': args.n,
                'number_of_samples_prior': args.number_of_samples_prior,
                'prior_path': args.prior_path,
                'prior_type': args.prior_type,
                'prior_scale': args.prior_scale,
                'random_state': args.random_state,
                'tune': args.tune,
                'wandb': args.wandb,
                'weight_decay': args.weight_decay,
            }
        )
    
    ce = torch.nn.CrossEntropyLoss()
    num_heads = 4
    checkpoint = torch.load(f'{args.prior_path}/resnet50_ssl_prior_model.pt', map_location=torch.device('cpu'))
    model = torchvision.models.resnet50() # Define model
    model.fc = torch.nn.Identity() # Get the classification head off
    model.load_state_dict(checkpoint) # Load the pretrained backbone weights
    model.fc = torch.nn.Linear(in_features=2048, out_features=num_heads, bias=True) # Put the proper classification head back
    model.to(device)
    
    if args.prior_type == 'nonlearned':
        criterion = losses.CustomCELoss(ce)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    elif args.prior_type == 'adapted':
        loc = torch.load(f'{args.prior_path}/resnet50_ssl_prior_mean.pt')
        loc = torch.cat((loc, torch.zeros((2048*num_heads)+num_heads)))
        criterion = losses.MAPAdaptationCELoss(ce, loc.cpu(), args.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=0.0, nesterov=True)
    elif args.prior_type == 'learned':
        loc = torch.load(f'{args.prior_path}/resnet50_ssl_prior_mean.pt')
        cov_factor = torch.load(f'{args.prior_path}/resnet50_ssl_prior_covmat.pt')
        cov_factor = args.prior_scale * (cov_factor[:args.number_of_samples_prior]) # Scale the low rank covariance
        cov_diag = torch.load(f'{args.prior_path}/resnet50_ssl_prior_variance.pt')
        cov_diag = args.prior_scale * cov_diag + args.prior_eps # Scale the variance
        criterion = losses.GaussianPriorCELossShifted(ce, loc.cpu(), cov_factor.t().cpu(), cov_diag.cpu())
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        raise NotImplementedError(f'The specified prior type \'{args.prior_type}\' is not implemented.')
        
    steps = 6000
    epochs = int(steps/len(train_loader_shuffled))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader_shuffled))

    columns = ['epoch', 'train_acc', 'train_loss', 'train_nll', 'train_prior', 'val_or_test_acc', 'val_or_test_loss', 'val_or_test_nll', 'val_or_test_prior']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(epochs):
        
        lrs = utils.train_one_epoch(model, criterion, optimizer, scheduler, train_loader_shuffled)
        train_loss, train_nll, train_prior, train_acc = utils.evaluate(model, criterion, train_loader, 'auroc', num_heads)
        
        if args.tune or epoch == epochs-1:
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = utils.evaluate(model, criterion, val_or_test_loader, 'auroc', num_heads)
        else:
            val_or_test_loss, val_or_test_nll, val_or_test_prior, val_or_test_acc = 0.0, 0.0, 0.0, 0.0
                            
        # Append evaluation metrics to DataFrame
        row = [epoch, train_acc, train_loss, train_nll, train_prior, val_or_test_acc, val_or_test_loss, val_or_test_nll, val_or_test_prior]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])
        
        if args.wandb:
            wandb.log({
                'epoch': epoch, 
                'train_acc': train_acc, 
                'train_loss': train_loss, 
                'train_nll': train_nll, 
                'train_prior': train_prior,
                'val_or_test_acc': val_or_test_acc, 
                'val_or_test_loss': val_or_test_loss, 
                'val_or_test_nll': val_or_test_nll, 
                'val_or_test_prior': val_or_test_prior,
            })
        
        model_history_df.to_csv(f'{args.experiments_path}/{args.model_name}.csv')
        
    if args.save:
        torch.save(model.state_dict(), f'{args.experiments_path}/{args.model_name}.pth')