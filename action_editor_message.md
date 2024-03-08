Hello Action Editor,
We were wonder if it would be acceptable for us to submit a revison as Reviewers have not been assigned yet.
After submitting our paper, we found a mistake in Shwartz-Ziv et al. (2022)'s implementation of *LearnedPriorLR* which was brought over into our code.
The mistake, which is detailed below, changes the shape of the prior covariance as the covariance scaling factor is increased.
The corrected *LearnedPriorLR* results show only a small increase in performance at some dataset sizes but fixing the mistake changes our conclusions about the alignment between train and test loss landscapes.

## Details
In Shwartz-Ziv et al.'s `load_prior()` function, 1) the prior variance is scaled by the covariance scaling factor and prior epsilon is added, $\Sigma_{\text{diag}} = \lambda \Sigma_{\text{diag}} + \epsilon I$ ([see line 37](https://github.com/hsouri/BayesianTransferLearning/blob/2bb409a25ab5154ed1fa958752c54842e34e9087/priorBox/sghmc/utils.py#L40)) and 2) if `number_of_samples_prior > 0` ($k > 0$) and `config['scale_low_rank'] is True` then the low-rank prior covariance factor is scaled by the covariance scaling factor, $Q = \lambda Q$ ([see line 40](https://github.com/hsouri/BayesianTransferLearning/blob/2bb409a25ab5154ed1fa958752c54842e34e9087/priorBox/sghmc/utils.py#L40)).
Step 2 here is incorrect. The low-rank prior covariance factor should be scaled by the square root of the covariance scaling factor, $Q = \sqrt{\lambda} Q$, because the covariance is constructed as $\Sigma = QQ^T + \Sigma_{\text{diag}}$.

###### Code from https://github.com/hsouri/BayesianTransferLearning/blob/2bb409a25ab5154ed1fa958752c54842e34e9087/priorBox/sghmc/utils.py
```python
37          variance = config['prior_scale'] * variance + config['prior_eps']
38          if number_of_samples_prior > 0:
39              if config['scale_low_rank']:
40                  cov_mat_sqrt = config['prior_scale'] * (cov_factor[:number_of_samples_prior])
41              else:
42                  cov_mat_sqrt = cov_factor[:number_of_samples_prior]
```

PyTorch's `torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(loc, cov_factor, cov_diag, validate_args=None)` class creates a multivariate normal distribution with covariance matrix having a low-rank form parameterized by cov_factor and cov_diag:

```python
covariance_matrix = cov_factor @ cov_factor.T + cov_diag
```

([see PyTorch documentation](https://pytorch.org/docs/stable/distributions.html#lowrankmultivariatenormal)).

Multiplying the low-rank prior covariance factor by the covariance scaling factor, $Q = \lambda Q$, results in scaling the low-rank covariance $\Sigma_{\text{LR}}$ exponentially more than the diagonal covariance $\Sigma_{\text{diag}}$, $\Sigma = \lambda \Sigma_{\text{diag}} + \lambda^2 \Sigma_{\text{LR}}$.

###### Code from https://github.com/hsouri/BayesianTransferLearning/blob/2bb409a25ab5154ed1fa958752c54842e34e9087/priorBox/sghmc/losses.py
```python
 2  from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
...
27          means = params['mean']
28          variance = params['variance']
29          cov_mat_sqr = params['cov_mat_sqr']
30          # Computes the Gaussian prior log-density.
31          self.mvn = LowRankMultivariateNormal(means, cov_mat_sqr.t(), variance)
```
