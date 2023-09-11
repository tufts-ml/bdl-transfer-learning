import os
import copy
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class ImageDataset2D(Dataset):
    def __init__(self, df, mean_and_std=None):
        self.path = df.path.to_list()
        self.label = df.label.to_list()
        if mean_and_std == None: self.mean_and_std = self.calc_mean_and_std()
        else: self.mean_and_std = mean_and_std

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        image = read_image(self.path[index]).float()
        return self.transform(image, normalize=True)[None,:,:,:], self.label[index]

    def calc_mean_and_std(self):
        running_mean, running_std = torch.zeros(3), torch.zeros(3)
        total_pixels = 0

        for path in self.path:
            image = read_image(path).float()
            image = self.transform(image)
            running_mean += torch.sum(image, dim=(1, 2))
            total_pixels += image.shape[1] * image.shape[2]
            
        mean = running_mean/total_pixels

        for path in self.path:
            image = read_image(path).float()
            image = self.transform(image)
            running_std += torch.sum((image - mean[:,None,None]) ** 2, dim=(1, 2))
        
        std = torch.sqrt(running_std/total_pixels)
        
        return tuple(mean.tolist()), tuple(std.tolist())

    def transform(self, item, normalize=False):
        transform_list = [transforms.Lambda(lambda x: x/255.0),
                          transforms.Resize((224, 224))]
        
        if normalize:
            transform_list.append(transforms.Normalize(mean=self.mean_and_std[0], std=self.mean_and_std[1]))

        transform = transforms.Compose(transform_list)
        return transform(item) 
    
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.concat(images, dim=0)
    return images, torch.tensor(labels).squeeze()

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def train_one_epoch(model, prior_params, criterion, optimizer, scheduler, dataloader, device):
    
    model.train()
    
    running_loss = 0.0
    lrs = list()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):     
        
        if device.type == 'cuda':
            inputs, targets = inputs.to(device), targets.to(device)
            
        if isinstance(prior_params, dict):
            optimizer.zero_grad()
            outputs = model(inputs)
            params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
            params = params[:prior_params['mean'].shape[0]].cpu()
            metrices = criterion(outputs, targets, N=prior_params['mean'].shape[0], params=params)
            metrices['loss'].backward()
            optimizer.step()
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
            running_loss += (len(inputs)/len(dataloader.dataset))*metrices['nll'].item()
        else:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.reshape(targets.shape[0])
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()
            running_loss += (len(inputs)/len(dataloader.dataset))*loss.item()

    return lrs

def evaluate(model, prior_params, criterion, dataloader, device):
    
    model.eval()
    
    running_loss = 0.0
    target_list, output_list = list(), list()
    
    if isinstance(prior_params, dict):
        params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()]))
        params = params[:prior_params['mean'].shape[0]].cpu()
        
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            if device.type == 'cuda':
                inputs, targets = inputs.to(device), targets.to(device)
        
            if isinstance(prior_params, dict):
                outputs = model(inputs)
                metrices = criterion(outputs, targets, N=prior_params['mean'].shape[0], params=params)
                running_loss += (len(inputs)/len(dataloader.dataset))*metrices['nll'].item()
            else:
                outputs = model(inputs)
                targets = targets.reshape(targets.shape[0])
                loss = criterion(outputs, targets)
                running_loss += (len(inputs)/len(dataloader.dataset))*loss.item()
            
            if device.type == 'cuda':
                targets, outputs = targets.cpu(), outputs.cpu()
                
            for output, target in zip(outputs, targets):
                target_list.append(target.numpy().astype(int))
                output_list.append(output.numpy())

    return running_loss, target_list, output_list

def bayesian_model_average(model, prior_params, criterion, dataloader, device, path):
    
    outputs_list = list()

    # Append outputs from current model
    loss, targets, outputs = evaluate(model, prior_params, criterion, dataloader, device)
    outputs_list.append(outputs)
    
    # Append outputs from previous models
    model = copy.deepcopy(model).to(device)
    
    for file in os.listdir(path):
        if not re.search('.pt$', file):
            continue
            
        model.load_state_dict(torch.load(os.path.join(path, file)))
        
        loss, targets, outputs = evaluate(model, prior_params, criterion, dataloader, device)
        outputs_list.append(outputs)
        
    return np.mean(outputs_list, axis=0)
