import torch
import torch.nn as nn

from model.bnn import *

class Bayesian_KL_Loss(nn.Module):
    """
    Calculate the KL divergence between the prior and posterior distribution of the model.
    """
    def __init__(self):
        super(Bayesian_KL_Loss, self).__init__()

    def forward(self, model):
        device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
        
        kl = torch.Tensor([0]).to(device)
        kl_sum = torch.Tensor([0]).to(device)
        n = torch.Tensor([0]).to(device)

        for m in model.modules() :
            if isinstance(m, (BayesLinear, BayesParameter)):
                kl = _Gaussian_kl_loss(m.weight_mu, m.weight_sigma, m.prior_mu, m.prior_sigma)
                kl_sum += kl
                n += len(m.weight_mu.view(-1))

                if m.bias :
                    kl = _Gaussian_kl_loss(m.bias_mu, m.bias_sigma, m.prior_mu, m.prior_sigma)
                    kl_sum += kl
                    n += len(m.bias_mu.view(-1))
                
        return kl_sum/n


def _Gaussian_kl_loss(mu1, sigma1, mu2, sigma2) :
    """
    Calculate the KL divergence between two Gaussian distribtuion.

    Args:
        mu1 (Float) : mean of normal distribution.
        sigma1 (Float): standard deviation of normal distribution.
        mu2 (Float): mean of normal distribution.
        sigma2 (Float): standard deviation of normal distribution.
   
    """
    kl = torch.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2 * sigma2**2) - 0.5
    return kl.sum()