import os
import copy
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torchmetrics

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
class ImageDataset2D(Dataset):
    def __init__(self, df, transform_images=True, mean_and_std=None):
        self.path = df.path.to_list()
        self.label = df.label.to_list()
        self.transform_images = transform_images
        if mean_and_std == None: self.mean_and_std = self.calc_mean_and_std()
        else: self.mean_and_std = mean_and_std
        #if self.transform_images: self.image = [self.transform(read_image(path).float(), normalize=True) for path in self.path]
        #else: self.image = [read_image(path).float() for path in self.path]

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        image = read_image(self.path[index]).float()
        if self.transform_images: image = self.transform(image, normalize=True)
        return image[None, :, :, :], self.label[index]
        #return self.image[index][None, :, :, :], self.label[index]

    def calc_mean_and_std(self):
        c, w, h = read_image(self.path[0]).float().shape # Gets number of channels from first image
        running_mean, running_std = torch.zeros(c), torch.zeros(c)
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
            running_std += torch.sum((image - mean[:, None, None]) ** 2, dim=(1, 2))
        
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
    # TODO: Don't hardcode number of classes
    auc = torchmetrics.AUROC(task='multiclass', num_classes=4, average='macro')
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
            running_auc += len(inputs)/len(dataloader.dataset)*auc(torch.softmax(outputs, dim=1).detach(), targets.detach()).item()

    return running_loss, running_nll, running_prior, running_auc