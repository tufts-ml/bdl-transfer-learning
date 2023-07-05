import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms

def makedir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class ImageDataset(Dataset):
    def __init__(self, df, mu=(0.4914, 0.4822, 0.4465), sigma=(0.247, 0.243, 0.261)):
        self.mu, self.sigma = mu, sigma
        self.path = df.path.to_list()
        self.label = df.label.to_list()
        
    def __len__(self):
        return len(self.path)

    def __getitem__(self, index):
        image = read_image(self.path[index]).float()
        return self.transform(image/255)[None,:,:,:], self.label[index]
    
    def transform(self, item):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(self.mu, self.sigma),
        ])
        return transform(item)
    
def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.concat(images, dim=0)
    return images, torch.tensor(labels).squeeze()

def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]

def update_params(model, device, datasize, lr, epoch, weight_decay=1e-5, alpha=0.9, temperature=1.0/50000):
    for p in model.parameters():
        if not hasattr(p, 'buf'):
            p.buf = torch.zeros(p.size()).to(device)
        d_p = p.grad.data
        d_p.add_(p.data, alpha=weight_decay)
        buf_new = (1-alpha)*p.buf - lr*d_p
        if (epoch%50)+1>45:
            eps = torch.randn(p.size()).to(device)
            buf_new += (2.0*lr*alpha*temperature/datasize)**.5*eps
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

def train_one_epoch(model, prior_params, device, criterion, lr_scheduler, dataloader, epoch):
    
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
        update_params(model, device, len(dataloader.dataset), lr, epoch)
        
        running_loss += metrices['nll'].item()

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

            running_loss += metrices['nll'].item()

    return running_loss, target_list, output_list