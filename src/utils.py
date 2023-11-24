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
from torchvision.datasets.utils import download_and_extract_archive
# Importing our custom module(s)
import folds

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_oxford_pets_datasets(root, n, tune=True, random_state=42):
    if not os.path.exists(os.path.join(root, 'annotations/')):
      URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
      download_and_extract_archive(URL, root)
    if not os.path.exists(os.path.join(root, 'images/')):
      URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
      download_and_extract_archive(URL, root)
    # Read annotations into dataframes 
    df_trainval = pd.read_csv(os.path.join(root, 'annotations/trainval.txt'),sep=' ',names=['image','id','species','breed_id'])
    n = int(n/df_trainval.id.max()) # number of images per class
    df_test = pd.read_csv(os.path.join(root, 'annotations/test.txt'),sep=' ',names=['image','id','species','breed_id'])
    # Add file paths to dataframe 
    df_trainval['path'] = df_trainval['image'].apply(lambda image: os.path.join(root, 'images/{}.jpg'.format(image)))
    df_test['path'] = df_test['image'].apply(lambda image: os.path.join(root, 'images/{}.jpg'.format(image)))
    if tune:
      df_train_sampled = df_trainval.groupby('id').apply(lambda x: x.sample(n=n, random_state=random_state)[:int(4/5*n)]).reset_index(drop = True)
      df_val_or_test_sampled = df_trainval.groupby('id').apply(lambda x: x.sample(n=n, random_state=random_state)[int(4/5*n):]).reset_index(drop = True)
    else:
      df_train_sampled = df_trainval.groupby('id').apply(lambda x: x.sample(n=n, random_state=random_state)[:int(4/5*n)]).reset_index(drop = True)
      df_val_or_test_sampled = df_test
    
    # For Oxford Pets we resize images before taking normalizing since images are different sizes
    transform = transforms.Resize(size=(256,256))
    # Some of the images have 4 channels RGBA. We ignore the last channel.
    imgs_train = []
    for path in df_train_sampled.path:
        temp = transform(read_image(path).float()) / 255
        temp = temp[:3,:,:] if temp.shape[0] == 4 else temp
        imgs_train.append(temp)
    sampled_train_images = torch.stack(imgs_train)
    imgs_val_or_test = []
    for path in df_val_or_test_sampled.path:
        temp = transform(read_image(path).float()) / 255
        temp = temp[:3,:,:] if temp.shape[0] == 4 else temp
        imgs_val_or_test.append(temp)
    sampled_val_or_test_images = torch.stack(imgs_val_or_test)
    
    #sampled_train_images = torch.stack([transform(read_image(path).float() / 255) for path in df_train_sampled.path])
    sampled_train_labels = torch.tensor([label-1 for label in df_train_sampled.id]).squeeze()
    train_mean = torch.mean(sampled_train_images, axis=(0, 2, 3))
    train_std = torch.std(sampled_train_images, axis=(0, 2, 3))
    #sampled_val_or_test_images = torch.stack([transform(read_image(path).float() / 255) for path in df_val_or_test_sampled.path])
    sampled_val_or_test_labels = torch.tensor([label-1 for label in df_val_or_test_sampled.id]).squeeze()

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.RandomCrop(size=(224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    val_or_test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.CenterCrop(size=(224, 224)),
    ])

    # Create Oxford Pets datasets
    augmented_train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, train_transform)
    train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, val_or_test_transform)
    val_or_test_dataset = CIFAR10(sampled_val_or_test_images, sampled_val_or_test_labels, val_or_test_transform)
    return augmented_train_dataset, train_dataset, val_or_test_dataset

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
        
    # Load CIFAR-10 datasets
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    # Sample int(n/10) datapoints per class from CIFAR-10 training dataset
    class_indices = {cifar10_label: [idx for idx, (image, label) in enumerate(cifar10_train_dataset) if label == cifar10_label] for cifar10_label in range(10)}
    sampled_class_indices = np.array([random_state.choice(indices, int(n/10), replace=False) for label, indices in class_indices.items()]).flatten()
    shuffled_sampled_class_indices = random_state.choice(sampled_class_indices, len(sampled_class_indices), replace=False)
    if tune:
        train_indices = shuffled_sampled_class_indices[:int(4/5*n)]
        val_or_test_indices = shuffled_sampled_class_indices[int(4/5*n):]
        val_or_test_dataset = [cifar10_train_dataset[index] for index in val_or_test_indices]
    else:
        train_indices = shuffled_sampled_class_indices
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
    print(device)
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