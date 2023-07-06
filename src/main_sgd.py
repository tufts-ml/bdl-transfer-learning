import argparse
import os
import ast
import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

from evaluation_metrics import *
from folds import *
from utils import *

# python bdl-transfer-learning/src/main_sgd.py --checkpoints_dir='/cluster/home/eharve06/bdl-transfer-learning/checkpoints' --prior_dir='/cluster/home/eharve06/resnet50_ssl_prior'

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--checkpoints_dir', type=str, default=None, required=True, help='directory to save checkpoints (default: None)')
    parser.add_argument('--data_dir', type=str, default='/cluster/tufts/hugheslab/eharve06/HAM10000', help='directory to dataset (default: "/cluster/tufts/hugheslab/eharve06/HAM10000")')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
    parser.add_argument('--lr_0', type=float, default=0.5, help='number of epochs to train (default: 0.5)')
    parser.add_argument('--prior_dir', type=str, default=None, help='directory to saved priors (default: None)')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
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
    train_loader_shuffled = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn)
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

    num_batch = math.ceil(len(train_loader.dataset)/args.batch_size)
    T = args.epochs*num_batch # Total number of iterations
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr_0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T)

    columns = ['epoch', 'train_loss', 'train_auroc', 
               'val_loss', 'val_auroc', 'test_loss', 
               'test_auroc','lrs']
    model_history_df = pd.DataFrame(columns=columns)
    
    for epoch in range(args.epochs):
        
        lrs = train_one_epoch(model, None, criterion, optimizer, scheduler, train_loader_shuffled, device)
        
        train_loss, train_targets, train_outputs = evaluate(model, None, criterion, train_loader, device)
        val_loss, val_targets, val_outputs = evaluate(model, None, criterion, val_loader, device)
        test_loss, test_targets, test_outputs = evaluate(model, None, criterion, test_loader, device)
        
        # Calculate AUROCs
        train_auroc = get_auroc(to_categorical(train_targets, num_classes), train_outputs)
        val_auroc = get_auroc(to_categorical(val_targets, num_classes), val_outputs)
        test_auroc = get_auroc(to_categorical(test_targets, num_classes), test_outputs)
                    
        # Append evaluation metrics to DataFrame
        row = [epoch+1, train_loss, train_auroc, val_loss, 
               val_auroc, test_loss, test_auroc, lrs]
        model_history_df.loc[epoch] = row
        print(model_history_df.iloc[epoch])

        model_history_df.to_csv('{}/model_history.csv'.format(args.checkpoints_dir))