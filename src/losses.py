from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import torch
import torch.nn as nn

class CustomCELoss(nn.Module):
    def __init__(self, ce):
        super().__init__()
        self.ce = ce
        self.number_of_params = 0

    def forward(self, logits, targets, N=None, params=None):
        nll = self.ce(logits, targets)
        matrices = {'loss': nll, 'nll': nll, 'prior': torch.tensor(0.0)}
        return matrices

class GaussianPriorCELossShifted(nn.Module):    
    def __init__(self, ce, loc, cov_factor, cov_diag):
        super().__init__()
        self.ce = ce
        self.number_of_params = loc.shape[0]
        self.mvn = LowRankMultivariateNormal(loc=loc, cov_factor=cov_factor, cov_diag=cov_diag)
    
    def log_prob(self, params):
        return self.mvn.log_prob(params)

    def forward(self, logits, targets, N, params):
        nll = self.ce(logits, targets)
        log_prior_value = self.log_prob(params).sum() / N
        log_prior_value = torch.clamp(log_prior_value, min=-1e20, max=1e20)
        ne_en = nll - log_prior_value # Negative energy
        matrices = {'loss': ne_en, 'nll': nll, 'prior': log_prior_value}
        return matrices
    
class MAPAdaptationCELoss(nn.Module):
    def __init__(self, ce, loc, weight_decay):
        super().__init__()
        self.ce = ce
        self.loc = loc
        self.number_of_params = loc.shape[0]
        self.weight_decay = weight_decay

    def forward(self, logits, targets, N, params):
        nll = self.ce(logits, targets)
        regularizer = self.weight_decay * torch.norm(params - self.loc)**2
        regularizer = torch.clamp(regularizer, min=-1e20, max=1e20)
        loss = nll + regularizer
        matrices = {'loss': loss, 'nll': nll, 'prior': regularizer}
        return matrices