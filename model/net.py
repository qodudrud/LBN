import torch
import torch.nn as nn
import numpy as np

import model.bnn as bnn
from model.bnn_loss import *


class ParameterLayer(nn.Module):
    def __init__(self, n_param):
        super(BayesParameter, self).__init__()
        self.n_param = n_param
        self.bias = False
        
        # Define the parameters of BNN
        # Note that the values of parameters will be assigned at the fuction 'reset_parameters'
        self.weight = nn.Parameter(torch.Tensor(n_param))
                
        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """ Assign the value of parameters, i.e., theta = (w_mu, w_sigma),

            weight_mu <- He Uniform initialization, i.e., w ~ U(-sqrt[6/input_dim, sqrt[6/input_dim]])
            weight_rho = self.posterior_rho_init

        """
        stdv =np.sqrt(6.0 / self.weight_mu.size(0))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_rho.data.fill_(self.posterior_rho_init)
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))

    def forward(self, x):
        """Computes the feedforwad operation with the sampled weights and biases
            if the model is frozen:
                w = w_mu
            else:
                w = w_mu + log (1 + exp(w_rho)) * w_eps
                    where w_eps ~ N(0, 1)
    
        Args:
            x (tensor) : input

        Returns:

        """
        if self.is_frozen:
            weight = self.weight_mu
        else:
            self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + self.weight_sigma * torch.randn_like(self.weight_sigma)
            
        return weight


class LNN_Drift(nn.Module):
    def __init__(self, opt):
        super(LNN_Drift, self).__init__()

        self.n_layer = opt.n_layer
        self.dim = opt.dim     # state dimension
        self.input_dim = opt.input_dim

        self.dt = torch.tensor(opt.time_step)
        self.device = opt.device

        self.force_model = nn.Sequential()

        if self.n_layer == 1:
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(opt.input_dim, opt.dim))
            setattr(self, "layer1", tmp)
            self.force_model.add_module("layer1", tmp)
        
        else:
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(opt.input_dim, opt.n_hidden))
            tmp.add_module("elu", nn.ELU(inplace=True))
            setattr(self, "layer1", tmp)
            self.force_model.add_module("layer1", tmp)
            for i in range(opt.n_layer - 2):
                tmp = nn.Sequential()
                tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))
                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer%d" % (i + 2), tmp)
                self.force_model.add_module("layer%d" % (i + 2), tmp)
            
            tmp = nn.Sequential()
            tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.dim))
            setattr(self, "layer%d" % (opt.n_layer), tmp)
            self.force_model.add_module("layer%d" % (opt.n_layer), tmp)
        

    def forward(self, x):
        loc = self.force_model(x)
        return loc * self.dt



class LNN_Diff(nn.Module):
    def __init__(self, opt):
        super(LNN_Diff, self).__init__()

        self.dim = opt.dim     # space dimension
        self.input_dim = opt.input_dim
        
        self.dt = torch.tensor(opt.time_step)
        self.device = opt.device
        self.diff_mode = opt.diff_mode

        if opt.diff_mode == 'homogeneous':
            self.tril_ind = torch.tril_indices(self.dim, self.dim)

            stdv = np.sqrt(2.0 / (self.input_dim  + int(opt.dim * (opt.dim+1)/2)))
            self.tril_param = nn.Parameter(torch.Tensor(int(opt.dim * (opt.dim+1)/2))).data.normal_(0, stdv)


        elif opt.diff_mode == 'inhomogeneous':
            self.n_layer = opt.n_layer
            self.tril_ind = torch.tril_indices(self.dim, self.dim)

            self.diff_model = nn.Sequential()

            if self.n_layer == 1:
                tmp = nn.Sequential()
                tmp.add_module("fc", nn.Linear(opt.input_dim, int(opt.dim * (opt.dim+1)/2)))
                setattr(self, "layer1", tmp)
                self.diff_model.add_module("layer1", tmp)

            else:
                tmp = nn.Sequential()
                tmp.add_module("fc", nn.Linear(opt.input_dim, opt.n_hidden))
                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer1", tmp)
                self.diff_model.add_module("layer1", tmp)
                for i in range(opt.n_layer - 2):
                    tmp = nn.Sequential()
                    tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))
                    tmp.add_module("elu", nn.ELU(inplace=True))
                    setattr(self, "layer%d" % (i + 2), tmp)
                    self.diff_model.add_module("layer%d" % (i + 2), tmp)
                
                tmp = nn.Sequential()
                tmp.add_module("fc", nn.Linear(opt.n_hidden, int(opt.dim * (opt.dim+1)/2)))
                setattr(self, "layer%d" % (opt.n_layer), tmp)
                self.diff_model.add_module("layer%d" % (opt.n_layer), tmp)
            

    def forward(self, x):
        if self.diff_mode == 'homogeneous':
            scale_tril = torch.zeros(self.dim, self.dim).to(self.device)
            scale_tril[self.tril_ind[0], self.tril_ind[1]] = self.tril_param 

            diffusion = torch.matmul(scale_tril, scale_tril.T)[self.tril_ind[0], self.tril_ind[1]] * self.dt

        elif self.diff_mode == 'inhomogeneous':
            scale_tril = torch.zeros(x.shape[0], x.shape[1], self.dim, self.dim).to(self.device)
            scale_tril[:, :, self.tril_ind[0], self.tril_ind[1]] = self.diff_model(x)

            diffusion = torch.einsum('nbik,nbjk->nbij', scale_tril, scale_tril)[:, :, self.tril_ind[0], self.tril_ind[1]] * self.dt

        return diffusion

    
    def diff_mat(self, x):
        if self.diff_mode == 'homogeneous':
            scale_tril = torch.zeros(self.dim, self.dim).to(self.device)
            scale_tril[self.tril_ind[0], self.tril_ind[1]] = self.tril_param 

            diffusion = torch.matmul(scale_tril, scale_tril.T)

        elif self.diff_mode == 'inhomogeneous':
            scale_tril = torch.zeros(x.shape[0], x.shape[1], self.dim, self.dim).to(self.device)
            scale_tril[:, :, self.tril_ind[0], self.tril_ind[1]] = self.diff_model(x)
            
            diffusion = torch.einsum('nbik,nbjk->nbij', scale_tril, scale_tril)

        return diffusion


class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean =  mean
        self.std = std
    
    def forward(self, x):
        return x * self.std + self.mean
    

class Tril2Diff(nn.Module):
    def __init__(self, opt, diag_ind, tril_ind, diag_ind_tril):
        super().__init__()
        self.dim = opt.dim
        self.diff_mode = opt.diff_mode
        self.device = opt.device

        self.diag_ind = diag_ind
        self.tril_ind = tril_ind
        self.diag_ind_tril = diag_ind_tril
    
    def forward(self, x):
        if self.diff_mode == 'homogeneous':
            scale_tril = torch.zeros(self.dim, self.dim).to(self.device)

        elif self.diff_mode == 'inhomogeneous':
            diff_shape = list(x.size())[:-1]
            diff_shape += [self.dim, self.dim]
            
            scale_tril = torch.zeros(diff_shape).to(self.device)

        scale_tril[..., self.tril_ind[0], self.tril_ind[1]] = x
        scale_tril[..., self.diag_ind[0], self.diag_ind[1]] = torch.log1p(torch.exp(x[..., self.diag_ind_tril]) + 1e-12)

        return torch.einsum('...ik,...jk->...ij', scale_tril, scale_tril)


class LBNN_Drift(nn.Module):
    def __init__(self, opt):
        super(LBNN_Drift, self).__init__()
        self.device = opt.device

        self.n_layer = opt.n_layer
        self.n_blayer = opt.n_blayer

        self.dim = opt.dim     # state dimension
        self.input_dim = opt.input_dim

        self.dt = torch.tensor(opt.time_step).to(opt.device)
        self.mean, self.std = torch.tensor(opt.output_mean).to(opt.device), torch.tensor(opt.output_std).to(opt.device)

        self.force_model = nn.Sequential()

        if self.n_layer == 1:
            tmp = nn.Sequential()
            tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=opt.dim))
            setattr(self, "layer1", tmp)
            self.force_model.add_module("layer1", tmp)
            
        
        else:
            for i in range(1, opt.n_layer - opt.n_blayer + 1):
                tmp = nn.Sequential()
                if i == 1:
                    tmp.add_module("fc", nn.Linear(opt.input_dim, opt.n_hidden))
                else:
                    tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))

                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer%d" % (i), tmp)
                self.force_model.add_module("layer%d" % (i), tmp)
            
            for i in range(opt.n_layer - opt.n_blayer + 1, opt.n_layer):
                tmp = nn.Sequential()
                if i == 1:
                    tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=opt.n_hidden))
                else:
                    tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=opt.n_hidden))
                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer%d" % (i), tmp)
                self.force_model.add_module("layer%d" % (i), tmp)
            
            tmp = nn.Sequential()
            tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=opt.dim))
            setattr(self, "layer%d" % (opt.n_layer), tmp)
            self.force_model.add_module("layer%d" % (opt.n_layer), tmp)

        self.force_model.add_module("Denormalize", DeNormalize(self.mean, self.std))

    def forward(self, x):
        loc = self.force_model(x)
        return loc * self.dt



class LBNN_Diff(nn.Module):
    def __init__(self, opt):
        super(LBNN_Diff, self).__init__()

        self.device = opt.device

        self.dim = opt.dim     # space dimension
        self.input_dim = opt.input_dim
        
        self.dt = torch.tensor(opt.time_step).to(opt.device)
        self.mean, self.std = torch.tensor(opt.output_mean).to(opt.device), torch.tensor(opt.output_std).to(opt.device)

        self.diff_mode = opt.diff_mode

        self.posterior_mu_init = 0.
        self.posterior_rho_init = -5.

        self.diff_model = nn.Sequential()

        self.tril_ind = torch.tril_indices(self.dim, self.dim)
        self.diag_ind = torch.stack([torch.arange(self.dim), torch.arange(self.dim)])
        self.off_diag_ind = torch.tril_indices(self.dim, self.dim, -1)

        self.diag_ind_tril = [int(i*(i+1)/2 - 1) for i in range(1, self.dim+1)]

        if opt.diff_mode == 'homogeneous':
            self.n_layer = 1
            self.n_blayer = 1
            
            self.diff_model.add_module("bnn_param", bnn.BayesParameter(int(opt.dim * (opt.dim+1)/2)))
            # self.tril_mu = nn.Parameter(torch.ones(int(opt.dim * (opt.dim+1)/2)))
            # self.tril_rho = nn.Parameter(self.posterior_rho_init * torch.ones(int(opt.dim * (opt.dim+1)/2)))
            

        elif opt.diff_mode == 'inhomogeneous':
            self.n_layer = opt.n_layer
            self.n_blayer = opt.n_blayer
            
            if self.n_layer == 1:
                tmp = nn.Sequential()
                tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=int(opt.dim * (opt.dim+1)/2)))
                setattr(self, "layer1", tmp)
                self.diff_model.add_module("layer1", tmp)
            
            else:
                for i in range(1, opt.n_layer - opt.n_blayer + 1):
                    tmp = nn.Sequential()
                    if i == 1:
                        tmp.add_module("fc", nn.Linear(opt.input_dim, opt.n_hidden))
                    else:
                        tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))

                    tmp.add_module("elu", nn.ELU(inplace=True))
                    setattr(self, "layer%d" % (i), tmp)
                    self.diff_model.add_module("layer%d" % (i), tmp)
                
                for i in range(opt.n_layer - opt.n_blayer + 1, opt.n_layer):
                    tmp = nn.Sequential()
                    if i == 1:
                        tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=opt.n_hidden))
                    else:
                        tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=opt.n_hidden))
                    tmp.add_module("elu", nn.ELU(inplace=True))
                    setattr(self, "layer%d" % (i), tmp)
                    self.diff_model.add_module("layer%d" % (i), tmp)
                
                tmp = nn.Sequential()
                tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=int(opt.dim * (opt.dim+1)/2)))
                setattr(self, "layer%d" % (opt.n_layer), tmp)
                self.diff_model.add_module("layer%d" % (opt.n_layer), tmp)
            
        self.diff_model.add_module("Denormalize", DeNormalize(self.mean, self.std))
        self.diff_model.add_module("Tril2Diff", Tril2Diff(opt, self.diag_ind, self.tril_ind, self.diag_ind_tril))


    def forward(self, x):
        diffusion = self.diff_model(x)
        diffusion = torch.nan_to_num(diffusion)
        return diffusion[..., self.tril_ind[0], self.tril_ind[1]] * self.dt

    
    def diff_mat(self, x, freeze=False):
        return self.diff_model(x)


class LBNN_Drift2(nn.Module):
    def __init__(self, opt):
        super(LBNN_Drift, self).__init__()
        self.device = opt.device

        self.n_layer = opt.n_layer
        self.n_blayer = opt.n_blayer

        self.dim = opt.dim     # state dimension
        self.input_dim = opt.input_dim

        self.dt = torch.tensor(opt.time_step).to(opt.device)
        self.mean, self.std = torch.tensor(opt.output_mean).to(opt.device), torch.tensor(opt.output_std).to(opt.device)

        self.force_model = nn.Sequential()

        if self.n_layer == 1:
            tmp = nn.Sequential()
            tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=opt.dim))
            setattr(self, "layer1", tmp)
            self.force_model.add_module("layer1", tmp)
            
        
        else:
            for i in range(1, opt.n_layer - opt.n_blayer + 1):
                tmp = nn.Sequential()
                if i == 1:
                    tmp.add_module("fc", nn.Linear(opt.input_dim, opt.n_hidden))
                else:
                    tmp.add_module("fc", nn.Linear(opt.n_hidden, opt.n_hidden))

                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer%d" % (i), tmp)
                self.force_model.add_module("layer%d" % (i), tmp)
            
            for i in range(opt.n_layer - opt.n_blayer + 1, opt.n_layer):
                tmp = nn.Sequential()
                if i == 1:
                    tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.input_dim, output_dim=opt.n_hidden))
                else:
                    tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=opt.n_hidden))
                tmp.add_module("elu", nn.ELU(inplace=True))
                setattr(self, "layer%d" % (i), tmp)
                self.force_model.add_module("layer%d" % (i), tmp)
            
            tmp = nn.Sequential()
            tmp.add_module("bnn_fc", bnn.BayesLinear(input_dim=opt.n_hidden, output_dim=opt.dim))
            setattr(self, "layer%d" % (opt.n_layer), tmp)
            self.force_model.add_module("layer%d" % (opt.n_layer), tmp)

        self.force_model.add_module("Denormalize", DeNormalize(self.mean, self.std))

    def forward(self, x):
        loc = self.force_model(x)
        return loc * self.dt
