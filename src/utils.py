import numpy as np
import torch

def update_params(model, device, datasize, lr, epoch, weight_decay=0.0, alpha=0.9, temperature=1.0/50000):
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

# ###Cosine Annealing LR
# def adjust_learning_rate(epoch, batch_idx):
#     #eta_min = 0
#     rcounter = epoch*num_batch+batch_idx
#     lr = (lr_0) *(1 + math.cos(math.pi * rcounter / T)) / 2
#     return lr

def train(model, prior_params, device, criterion, lr_scheduler, dataloader, epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    lrs = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        if device.type == 'cuda':
            inputs, targets = inputs.to(device), targets.to(device)
            
        model.zero_grad()
        lr = lr_scheduler.adjust_learning_rate(epoch, batch_idx)
        lrs.append(lr)
        outputs = model(inputs)
        params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()])) ## Flatten all the parms to one array
        params = params[:prior_params['mean'].shape[0]].cpu()
        metrices = criterion(outputs, targets.data, N=prior_params['mean'].shape[0], params=params)
        metrices['loss'].backward()
        update_params(model, device, len(dataloader.dataset), lr, epoch)

        train_loss += metrices['nll'].item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx%100==0: 
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
    return lrs

def test(model, prior_params, device, criterion, dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    params = torch.flatten(torch.cat([torch.flatten(p) for p in model.parameters()])) ## Flatten all the parms to one array
    params = params[:prior_params['mean'].shape[0]].cpu()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if device.type == 'cuda':
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            metrices = criterion(outputs, targets.data, N=prior_params['mean'].shape[0], params=params)

            test_loss += metrices['nll'].item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(dataloader), correct, total,
    100. * correct.item() / total))
    return test_loss/len(dataloader), correct.item() / total