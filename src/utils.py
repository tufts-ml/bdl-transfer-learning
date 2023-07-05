import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import copy
import re

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

def update_params(model, device, datasize, lr, epoch, weight_decay, alpha, temperature):
    for p in model.parameters():
        if not hasattr(p, 'buf'):
            p.buf = torch.zeros(p.size()).to(device)
        d_p = p.grad.data
        d_p.add_(p.data, alpha=weight_decay)
        buf_new = (1-alpha)*p.buf - lr*d_p
        if (epoch%50)+1>45:
            eps = torch.randn(p.size()).to(device)
            buf_new += (2.0*lr*alpha*temperature/datasize)**0.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new
        
class CosineAnnealingLR():    
    def __init__(self, num_batch, T, M=4, lr_0=0.5):
        self.num_batch = num_batch # total number of iterations
        self.T = T # total number of iterations
        self.lr_0 = lr_0 # initial lr
        self.M = M # number of cycles

    def adjust_learning_rate(self, epoch, batch_idx):
        rcounter = epoch*self.num_batch + batch_idx
        cos_inner = np.pi * (rcounter % (self.T // self.M))
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * self.lr_0
        return lr

def train_one_epoch(model, prior_params, device, criterion, lr_scheduler, dataloader, epoch, args):
    
    model.train()
    
    running_loss = 0.0
    lrs = list()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):     
        
        if device.type == 'cuda':
            inputs, targets = inputs.to(device), targets.to(device)
            
        model.zero_grad()
        lr = lr_scheduler.adjust_learning_rate(epoch, batch_idx)
        lrs.append(lr)
        outputs = model(inputs)
        params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()])) ## Flatten all the parms to one array
        params = params[:prior_params['mean'].shape[0]].cpu()
        metrices = criterion(outputs, targets, N=prior_params['mean'].shape[0], params=params)
        metrices['loss'].backward()
        update_params(model, device, len(dataloader.dataset), lr, epoch, args.weight_decay, args.alpha, args.temperature)
        
        running_loss += (len(inputs)/len(dataloader.dataset))*metrices['nll'].item()

    return lrs

def evaluate(model, prior_params, device, criterion, dataloader):
    
    model.eval()
    
    running_loss = 0.0
    target_list, output_list = list(), list()
    
    params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()])) ## Flatten all the parms to one array
    params = params[:prior_params['mean'].shape[0]].cpu()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            
            if device.type == 'cuda':
                inputs, targets = inputs.to(device), targets.to(device)
                
            outputs = model(inputs)
            metrices = criterion(outputs, targets, N=prior_params['mean'].shape[0], params=params)
            
            if device.type == 'cuda':
                targets, outputs = targets.cpu(), outputs.cpu()
                
            for output, target in zip(outputs.cpu(), targets):
                target_list.append(target.numpy().astype(int))
                output_list.append(output.numpy())

            running_loss += (len(inputs)/len(dataloader.dataset))*metrices['nll'].item()

    return running_loss, target_list, output_list

def bayesian_model_average(model, prior_params, device, criterion, dataloader, path):
    
    outputs_list = list()

    # Append outputs from current model
    loss, targets, outputs = evaluate(model, prior_params, device, criterion, dataloader)
    outputs_list.append(outputs)
    
    # Append outputs from previous models
    model = copy.deepcopy(model).to(device)
    
    for file in os.listdir(path):
        if not re.search('.pt$', file):
            continue
            
        model.load_state_dict(torch.load(os.path.join(path, file)))
        
        loss, targets, outputs = evaluate(model, prior_params, device, criterion, dataloader)
        outputs_list.append(outputs)
        
    return outputs_list if len(outputs_list) == 1 else np.mean(outputs_list, axis=0)