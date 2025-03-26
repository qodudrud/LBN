import torch
import numpy as np

# Van der Pol oscilator model
def simulation_ULE_VdP(num_trjs, trj_len, dim, T, dt, sample_freq=10, seed=0, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    k = 2
    
    T_array = torch.ones(dim) * np.sqrt(2 * T)
    sqrt_Diffusion = torch.diag(T_array).to(device)
    
    dt2 = np.sqrt(dt)
    
    trj = torch.zeros(num_trjs, trj_len * sample_freq, 2*dim).to(device)
    
    x = torch.zeros(dim, num_trjs).to(device)
    v = torch.zeros(dim, num_trjs).to(device)
    
    for _ in range(10000):
        DriftForce = k * ( 1 - x**2) * v - x

        RanForce = torch.randn(dim, num_trjs, device=device)
        RanForce = sqrt_Diffusion @ RanForce
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2
    
    for it in range(trj_len * sample_freq):
        DriftForce = k * ( 1 - x**2) * v - x

        RanForce = torch.randn(dim, num_trjs, device=device)
        RanForce = sqrt_Diffusion @ RanForce
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2

        trj[:, it, :] = torch.cat((x, v), axis=0).T

    return trj[:, ::sample_freq]


def diff_mat(trajs, opt):
    T_array = torch.ones(opt.dim) * opt.Tc
    Diffusion = torch.diag(T_array)
    
    return Diffusion


def drift_force(trajs, opt):
    k = 2

    pos_trajs, vel_trajs = trajs[..., :opt.dim], trajs[..., opt.dim:]
    return k * ( 1 - pos_trajs**2) * vel_trajs - pos_trajs

