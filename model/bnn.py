import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

"""
Bayesian Neural Network (BNN) based on Bayes by Backprop. (reference; Weight Uncertainty in Neural Networks)

References:
    https://github.com/Harry24k/bayesian-neural-network-pytorch
"""

class BayesParameter(nn.Module):
    """
    Bayesian Parameter based on Bayes by Backprop.
    
    Args:
        n_param (int) : the number of parameters
        posterior_mu_init (float) : initial mu of the posterior distribution (default: 0.)
        posterior_rho_init (float) : initial rho of the posterior distribution (default: -5. -> initial sigma ~ 0.007 )
        prior_mu (float) : initial mu of the prior distribution (default: 0.)
        prior_sigma (float) : initial mu of the prior distribution (default: 0.1)
        is_frozen (bool) :  if set to True, the layer will return same results with same inputs (default: True)
    """
    def __init__(self, n_param, posterior_mu_init = 0., posterior_rho_init = -5., \
        prior_mu=0., prior_sigma=0.1, is_frozen=False):
        super(BayesParameter, self).__init__()
        self.n_param = n_param
        self.is_frozen = is_frozen
        self.bias = False
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        # Define the parameters of BNN
        # Note that the values of parameters will be assigned at the fuction 'reset_parameters'
        self.weight_mu = nn.Parameter(torch.Tensor(n_param))
        self.weight_rho = nn.Parameter(torch.Tensor(n_param))
        self.weight_sigma = None
                
        # initialize parameters
        self.reset_parameters()
    

    def reset_parameters(self):
        """ 
        Assign the value of parameters, i.e., theta = (w_mu, w_sigma),

            weight_mu <- He Uniform initialization, i.e., w ~ U(-sqrt[6/input_dim, sqrt[6/input_dim]])
            weight_rho = self.posterior_rho_init
            weight_sigma = log(1 + exp(weight_rho))
        """
        stdv = np.sqrt(6.0 / self.weight_mu.size(0))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(self.posterior_rho_init)
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))


    def freeze(self) :
        """
        If the model is frozen, the parameters are fixed to their mean values
        """  
        self.is_frozen = True

    def unfreeze(self):
        """
        If the model is ufrozen, the parameters are sampled from the (Gaussian) distributions
        """  
        self.is_frozen = False


    def forward(self, x):
        """
        Computes the feedforwad operation with the sampled weights and biases
            if the model is frozen:
                w = w_mu
            else:
                w = w_mu + log (1 + exp(w_rho)) * w_eps
                    where w_eps ~ N(0, 1)
    
        Args:
            x (tensor) : input

        Returns:
            w (tensor) : sampled weight

        """
        if self.is_frozen:
            weight = self.weight_mu
        else:
            self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            
        return weight


class BayesLinear(nn.Module):
    def __init__(self, input_dim, output_dim, posterior_mu_init = 0., posterior_rho_init = -5., \
        prior_mu=0., prior_sigma=0.1, bias=True, is_frozen=False):
        """
        Bayesian Linear layer based on Bayes by Backprop.
            we assume that the prior and posterior distributions are Gaussian distributions with mean = mu and std.=sigma.
    
        Args:
            input_dim (int) : input dimensions for the layer
            output_dim (int) : output dimensions for the layer
            posterior_mu_init (float) : initial mu of the posterior distribution (default: 0.)
            posterior_rho_init (float) : initial rho of the posterior distribution (default: -5. -> initial sigma ~ 0.07 )
            prior_mu (float) : initial mu of the prior distribution (default: 0.)
            prior_sigma (float) : initial mu of the prior distribution (default: 0.1)
            bias (bool) : if set to False, the layer will not learn an additive bias (default: True)
            is_frozen (bool) :  if set to True, the layer will return same results with same inputs (default: True)
        """  
        super(BayesLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.is_frozen = is_frozen
        
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        
        # Define the parameters of BNN
        # Note that the values of parameters will be assigned at the fuction 'reset_parameters'
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_rho = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_sigma = None
            
        if self.bias:
            self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
            self.bias_rho = nn.Parameter(torch.Tensor(output_dim))
            self.bias_sigma = None
            self.register_buffer('bias_eps', None)

        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
            self.register_parameter('bias_log_sigma', None)
            self.register_buffer('bias_eps', None)
        
        # initialize parameters
        self.reset_parameters()
    

    def reset_parameters(self):
        """ 
        Assign the value of parameters, i.e., theta = (w_mu, w_sigma, b_mu, b_sigma),

            weight_mu <- He Uniform initialization, i.e., w ~ U(-sqrt[6/input_dim, sqrt[6/input_dim]])
            weight_rho = self.posterior_rho_init
            bias_mu = 0
            bias_rho = self.posterior_rho_init

        """
        stdv =np.sqrt(6.0 / self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(self.posterior_rho_init)
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        if self.bias :
            self.bias_mu.data.zero_()
            self.bias_rho.data.fill_(self.posterior_rho_init)
            self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))


    def freeze(self) :
        """
        If the model is frozen, the parameters are fixed to their mean values

        """  
        self.is_frozen = True


    def unfreeze(self):
        """
        If the model is ufrozen, the parameters are sampled from the (Gaussian) distributions

        """  
        self.is_frozen = False


    def forward(self, x):
        """Computes the feedforwad operation with the sampled weights and biases
            if the model is frozen:
                w = w_mu, b = b_mu
            else:
                w = w_mu + log (1 + exp(w_rho)) * w_eps, 
                b = b_mu + log (1 + exp(b_rho)) * b_eps
                    where w_eps, b_eps ~ N(0, 1)
    
        Args:
            x (tensor) : input

        Returns:
            y (tensor) : output
            where y = x * w + b

        """
        if self.is_frozen:
            weight = self.weight_mu
        else:
            self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)        

        if self.bias:
            if self.is_frozen:
                bias = self.bias_mu            
            else:
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + self.bias_sigma * torch.randn_like(self.bias_sigma)
        else :
            bias = None
            
        return F.linear(x, weight, bias)