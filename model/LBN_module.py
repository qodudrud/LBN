import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy


class Langevin_from_LBN:
    def __init__(self, drift_model, diff_model, opt):
        self.system = opt.System
        self.dim = opt.dim

        # parameters of drift_ and diff_model should be in the same device with opt.device
        self.drift_model = drift_model
        self.diff_model = diff_model

        self.mean = opt.mean
        self.std = opt.std

        self.diff_mode = opt.diff_mode
        self.sample_freq = opt.sample_freq
        self.time_step = opt.time_step
        self.n_ensemble = opt.n_ensemble

        self.device = opt.device
        
    def simulation_from_model(self, num_trjs, trj_len, seed=0):
        np.random.seed(seed)
        torch.manual_seed(seed)

        drift_model, diff_model = self.drift_model, self.diff_model
        
        noises = torch.randn(self.dim, num_trjs, trj_len, device=self.device)
        
        dt = self.time_step
        dt2 = np.sqrt(dt)

        trj = torch.zeros(num_trjs, trj_len, self.dim).to(self.device)
        x = torch.zeros(self.dim, num_trjs).to(self.device)
        
        trj_ensemble = torch.zeros(num_trjs, trj_len, self.n_ensemble, self.dim).to(self.device)
        
        for it in range(trj_len):
            norm_x = ((x.T - self.mean)/self.std)
                    
            for j in range(self.n_ensemble):
                DriftForce = (drift_model.force_model(norm_x.to(self.device)) * self.std.to(self.device)).cpu().detach()
            
                diffusion_mat = (diff_model.diff_mat(norm_x.to(self.device)) * torch.outer(self.std, self.std).to(self.device)).cpu().detach()
                sqrt_diffusion = np.sqrt(2) * torch.FloatTensor(scipy.linalg.sqrtm(diffusion_mat))
                RanForce = sqrt_diffusion @ noises[..., it]

                trj_ensemble[:, it, j] = (x + (DriftForce.T * dt + RanForce * dt2)).T
            
            x = trj_ensemble[:, it].mean(axis=-2).T
                    
            trj[:, it, :] = x.T

        return trj, trj_ensemble
