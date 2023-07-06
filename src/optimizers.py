import math
import torch
from torch.optim import Optimizer

class SGHMC(Optimizer):
    def __init__(self, params, batch_size, datasize, device, alpha=0.9, lr=0.5, 
                 temperature=1.0/50000, weight_decay=5e-4):
        # Call the base Optimizer constructor
        defaults = dict(alpha=alpha, batch_size=batch_size, datasize=datasize, 
                        device=device, step=0, lr=lr, temperature=temperature, 
                        weight_decay=weight_decay)
        super(SGHMC, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            epoch = int(group['step']/math.ceil(group['datasize']/group['batch_size']))
            for p in group['params']:
                if p.grad is None:
                    continue
                if not hasattr(p, 'buf'):
                    p.buf = torch.zeros(p.size()).to(group['device'])
                d_p = p.grad.data
                d_p.add_(p.data, alpha=group['weight_decay'])
                buf_new = (1-group['alpha'])*p.buf - group['lr']*d_p
                if (epoch%50)+1>45:
                    eps = torch.randn(p.size()).to(group['device'])
                    buf_new += (2.0*group['lr']*group['alpha']*group['temperature']/group['datasize'])**0.5*eps
                p.data.add_(buf_new)
                p.buf = buf_new
        group['step'] += 1