from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
import torch
import torch.nn as nn

class CustomCELoss(nn.Module):
    """Wrapper for CrossEntropy that accepts N and params in forward function"""

    def __init__(self, ce):
        super().__init__()
        self.ce = ce

    def forward(self, logits, Y, N=None, params=None):
        nll = self.ce(logits, Y)
        matrices = {'loss': nll, 'nll': nll, 'prior': torch.Tensor([0])}
        return matrices

class GaussianPriorCELossShifted(nn.Module):
    """Scaled CrossEntropy + Gaussian prior"""

    def __init__(self, ce, params):
        super().__init__()
        self.ce = ce
        means = params['mean']
        variance = params['variance']
        cov_mat_sqr = params['cov_mat_sqr']
        # Computes the Gaussian prior log-density
        self.mvn = LowRankMultivariateNormal(means, cov_mat_sqr.t(), variance)
    
    def log_prob(self, params):
        return self.mvn.log_prob(params)

    def forward(self, logits, Y, N, params):
        nll = self.ce(logits, Y)
        log_prior_value = self.log_prob(params).sum() / N
        log_prior_value = torch.clamp(log_prior_value, min=-1e20, max=1e20)

        ne_en = nll - log_prior_value # Negative energy
        matrices = {'loss': ne_en, 'nll': nll, 'prior': log_prior_value}
        return matrices