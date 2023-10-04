import os
import numpy as np
# PyTorch
import torch
import torchvision
import torchmetrics

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class CIFAR10(torch.utils.data.Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (self.transform(self.X[index]), self.y[index]) if self.transform else (self.X[index], self.y[index])

def get_cifar10_datasets(root, n, random_state=42):
    
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(int(random_state))
    if not hasattr(random_state, 'rand'):
        raise ValueError('Not a valid random number generator')
        
    # Load CIFAR-10 datasets
    cifar10_train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True)
    cifar10_test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True)
    # Sample train and validation datasets from CIFAR-10 training dataset
    random_indices = random_state.choice(np.arange(len(cifar10_train_dataset)), n, replace=False)
    train_indices = random_indices[:int(4/5*n)] # Use 4/5 of training_samples for training
    val_indices = random_indices[int(4/5*n):] # Use 1/5 of training_samples for validation
    # Use entire CIFAR-10 testing dataset
    test_indices = range(len(cifar10_test_dataset))
    # Sample CIFAR10 training and validation datasets
    sampled_train_dataset = [cifar10_train_dataset[index] for index in train_indices]
    sampled_val_dataset = [cifar10_train_dataset[index] for index in val_indices]
    # Get channel mean and std from training data
    to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    sampled_train_images = torch.stack([to_tensor(image) for image, label in sampled_train_dataset])
    train_mean = torch.mean(sampled_train_images, axis=(0,2,3))
    train_std = torch.std(sampled_train_images, axis=(0,2,3))
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.RandomCrop(size=(244, 244)),
        torchvision.transforms.RandomHorizontalFlip(),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Normalize(mean=train_mean, std=train_std),
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.CenterCrop(size=(244, 244)),
    ])
    # Load X and y for each dataset
    sampled_train_images = torch.stack([to_tensor(image) for image, label in sampled_train_dataset])
    sampled_train_labels = torch.tensor([label for image, label in sampled_train_dataset])
    sampled_val_images = torch.stack([to_tensor(image) for image, label in sampled_val_dataset])
    sampled_val_labels = torch.tensor([label for image, label in sampled_val_dataset])
    test_images = torch.stack([to_tensor(image) for image, label in cifar10_test_dataset])
    test_labels = torch.tensor([label for image, label in cifar10_test_dataset])
    # Create CIFAR10 datasets
    train_dataset = CIFAR10(sampled_train_images, sampled_train_labels, train_transform)
    val_dataset = CIFAR10(sampled_val_images, sampled_val_labels, test_transform)
    test_dataset = CIFAR10(test_images, test_labels, test_transform)
    return train_dataset, val_dataset, test_dataset

def train_one_epoch(model, prior_params, criterion, optimizer, scheduler, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    model.train()
    
    lrs = list()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        if device.type == 'cuda':
            inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
        params = params[:prior_params['mean'].shape[0]].cpu()
        metrices = criterion(outputs, targets, N=len(dataloader.dataset), params=params)
        metrices['loss'].backward()
        optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
        
    return lrs

def evaluate(model, prior_params, criterion, dataloader):
    
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    #auc = torchmetrics.Accuracy(task='multiclass', num_classes=model.fc.out_features, average='macro')
    auc = torchmetrics.Accuracy(task='multiclass', num_classes=10, average='macro')
    model.eval()
    
    running_loss, running_nll, running_prior, running_auc = 0.0, 0.0, 0.0, 0.0
    
    params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
    params = params[:prior_params['mean'].shape[0]].cpu()
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            if device.type == 'cuda':
                inputs, targets = inputs.to(device), targets.to(device)
                
            outputs = model(inputs)
            metrices = criterion(outputs, targets, N=len(dataloader.dataset), params=params)
            
            running_loss += len(inputs)/len(dataloader.dataset)*metrices['loss'].item()
            running_nll += len(inputs)/len(dataloader.dataset)*metrices['nll'].item()
            running_prior += len(inputs)/len(dataloader.dataset)*metrices['prior'].item()
            running_auc += len(inputs)/len(dataloader.dataset)*auc(torch.softmax(outputs, dim=1).detach().cpu(), targets.detach().cpu()).item()

    return running_loss, running_nll, running_prior, running_auc