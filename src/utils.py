import os
import ast
import copy
import re
import numpy as np
import pandas as pd
# PyTorch
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchmetrics
# Importing our custom module(s)
import folds

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_ham10000_datasets(root, n, tune=True, random_state=42):
    # Load HAM10000 datasets (see HAM10000.ipynb to create labels.csv)
    ham10000_train_df = pd.read_csv(os.path.join(root, 'train/labels.csv'), index_col='lesion_id')
    ham10000_test_df = pd.read_csv(os.path.join(root, 'test/labels.csv'), index_col='lesion_id')
    ham10000_train_df.label = ham10000_train_df.label.apply(lambda item: ast.literal_eval(item))
    ham10000_test_df.label = ham10000_test_df.label.apply(lambda item: ast.literal_eval(item))
    # Randomly sample n datapoints from HAM10000 training DataFrame
    sampled_ham10000_train_df = ham10000_train_df.sample(n=n, random_state=random_state)
    if tune:
        # Create folds
        sampled_ham10000_train_df['Fold'] = folds.create_folds(sampled_ham10000_train_df, index_name='lesion_id', random_state=random_state)
        # Split folds
        train_df, val_or_test_df = folds.split_folds(sampled_ham10000_train_df)
    else:
        train_df = sampled_ham10000_train_df
        val_or_test_df = ham10000_test_df
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(lambda item: item/255),
    ])
    sampled_train_images = torch.stack([to_tensor(read_image(path).float()) for path in train_df.path])
    train_mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    train_std = torch.std(sampled_train_images, axis=(0, 2, 3))
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    val_or_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    sampled_train_images = torch.stack([to_tensor(read_image(path).float()) for path in train_df.path])
    sampled_train_labels = torch.tensor([label for label in train_df.label]).squeeze()
    sampled_val_or_test_images = torch.stack([to_tensor(read_image(path).float()) for path in val_or_test_df.path])
    sampled_val_or_test_labels = torch.tensor([label for label in val_or_test_df.label]).squeeze()
    # Create HAM10000 datasets
    augmented_train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, train_transform)
    train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, val_or_test_transform)
    val_or_test_dataset = CIFAR10(sampled_val_or_test_images, sampled_val_or_test_labels, val_or_test_transform)
    return augmented_train_dataset, train_dataset, val_or_test_dataset

class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.transform(self.X[index]), self.y[index]) if self.transform else (self.X[index], self.y[index])

def get_cifar10_datasets(root, n, tune=True, random_state=42):
    
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
        
    assert n%50 == 0 or n == 10, 'Invalid number of samples n={}'.format(n)
    # Load CIFAR-10 datasets
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    # Dictionary of labels and indices
    class_indices = {cifar10_label: [idx for idx, (image, label) in enumerate(cifar10_train_dataset) if label == cifar10_label] for cifar10_label in range(10)}
    shuffled_sampled_class_indices = {cifar10_label: random_state.choice(class_indices[cifar10_label], int(n/10), replace=False) for cifar10_label in class_indices.keys()}
    if tune:
        if n == 10:
            mask = random_state.choice(np.tile([True, True, True, True, False], reps=int(n/5))[:n], n, replace=False)
            train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[mask]
            val_or_test_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()[~mask]
        else:
            train_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][:int(4/50*n)] for cifar10_label in shuffled_sampled_class_indices.keys()}
            val_or_test_indices = {cifar10_label: shuffled_sampled_class_indices[cifar10_label][int(4/50*n):] for cifar10_label in shuffled_sampled_class_indices.keys()}
            train_indices = np.array(list(train_indices.values())).flatten()
            val_or_test_indices = np.array(list(val_or_test_indices.values())).flatten()
        print(val_or_test_indices)
        val_or_test_dataset = [cifar10_train_dataset[index] for index in val_or_test_indices]
    else:
        train_indices = np.array(list(shuffled_sampled_class_indices.values())).flatten()
        val_or_test_dataset = cifar10_test_dataset
    # Get channel mean and std from training data
    sampled_train_dataset = [cifar10_train_dataset[index] for index in train_indices]
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    sampled_train_images = torch.stack([to_tensor(image) for image, label in sampled_train_dataset])
    train_mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    train_std = torch.std(sampled_train_images, axis=(0, 2, 3))
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    val_or_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])
    # Load X and y for each dataset
    sampled_train_images = torch.stack([to_tensor(image) for image, label in sampled_train_dataset])
    sampled_train_labels = torch.tensor([label for image, label in sampled_train_dataset])
    sampled_val_or_test_images = torch.stack([to_tensor(image) for image, label in val_or_test_dataset])
    sampled_val_or_test_labels = torch.tensor([label for image, label in val_or_test_dataset])
    # Create CIFAR10 datasets
    augmented_train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, train_transform)
    train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, val_or_test_transform)
    val_or_test_dataset = CIFAR10(sampled_val_or_test_images, sampled_val_or_test_labels, val_or_test_transform)
    return augmented_train_dataset, train_dataset, val_or_test_dataset

def train_one_epoch(model, criterion, optimizer, scheduler, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    lrs = []
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        if device.type == 'cuda':
            inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
        params = params[:criterion.number_of_params].cpu()
        metrices = criterion(outputs, targets, N=len(dataloader.dataset), params=params)
        metrices['loss'].backward()
        optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
    return lrs

def evaluate(model, criterion, dataloader, metric='accuracy', num_classes=10):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.eval()    

    if metric == 'accuracy':
        acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes, average='macro')
    elif metric == 'auroc':
        acc = torchmetrics.AUROC(task='multiclass', num_classes=num_classes, average='macro')
    else:
        raise NotImplementedError('The specified metric \'{}\' is not implemented.'.format(metric))
    
    outputs_list, targets_list = [], []
    running_loss, running_nll, running_prior = 0.0, 0.0, 0.0
    
    params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
    params = params[:criterion.number_of_params].cpu()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            if device.type == 'cuda':
                inputs, targets = inputs.to(device), targets.to(device)
                
            outputs = model(inputs)
            metrices = criterion(outputs, targets, N=len(dataloader.dataset), params=params)
            
            running_loss += len(inputs)/len(dataloader.dataset)*metrices['loss'].item()
            running_nll += len(inputs)/len(dataloader.dataset)*metrices['nll'].item()
            running_prior += len(inputs)/len(dataloader.dataset)*metrices['prior'].item()
            
            if device.type == 'cuda':
                outputs, targets = outputs.cpu(), targets.cpu()
            
            for output, target in zip(outputs, targets):
                outputs_list.append(output)
                targets_list.append(target)
            
        outputs = torch.stack(outputs_list)
        targets = torch.stack(targets_list)
        running_acc = acc(torch.softmax(outputs, dim=1), targets).item()

    return running_loss, running_nll, running_prior, running_acc

def print_difference(model):
    model = model.cpu()
    pretrained_model = load_ViT()
    pretrained_weights = pretrained_model.state_dict()
    for name, param in model.named_parameters():
        if not 'heads.head' in name:
            # TODO: Assert that names are the same
            pretrained_size = pretrained_weights[name].shape
            slice_indices = tuple(slice(0, int(dim)) for dim in pretrained_size)
            print(torch.norm(param[slice_indices]-pretrained_weights[name]))

def load_ViT():
    # ViT-Base
    image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim = 224, 16, 12, 12, 768, 3072
    pretrained_model = torchvision.models.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
    )
    pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_weights = torchvision.models.ViT_B_16_Weights(pretrained_weights)
    pretrained_model.load_state_dict(pretrained_weights.get_state_dict(progress=True))
    return pretrained_model

def load_modified_ViT(hidden_dim_per_head=64, frozen=True):
    # ViT-Base
    image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim = 224, 16, 12, 12, 768, 3072
    pretrained_model = torchvision.models.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
    )
    pretrained_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    pretrained_weights = torchvision.models.ViT_B_16_Weights(pretrained_weights)
    pretrained_model.load_state_dict(pretrained_weights.get_state_dict(progress=True))
    pretrained_weights = pretrained_model.state_dict()
    # Modified ViT-Base
    modified_hidden_dim = hidden_dim_per_head*num_heads
    modified_model = torchvision.models.VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=modified_hidden_dim,
        mlp_dim=mlp_dim,
    )
    modified_weights = modified_model.state_dict()
    for (pretrained_name, pretrained_param), (modified_name, modified_param) in zip(pretrained_weights.items(), modified_weights.items()):
        # TODO: Assert that names are the same
        pretrained_size = pretrained_weights[pretrained_name].shape
        modified_size = modified_weights[modified_name].shape
        modified_param = torch.zeros(modified_size)
        slice_indices = tuple(slice(0, int(dim)) for dim in pretrained_size)
        modified_param[slice_indices] = pretrained_param
        modified_weights[pretrained_name] = modified_param 
    modified_model.load_state_dict(modified_weights)
    return modified_model