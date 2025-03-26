import torch
import numpy as np

e, k, T0 = 1, 2, 2

def simulation_inhomogeneous(num_trjs, trj_len, dim, T, dt, sample_freq=10, seed=0, device='cpu'):
    # reset the seed for generating trajectories
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dt2 = np.sqrt(dt)

    Drift = torch.zeros(dim, dim).to(device)    
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < dim - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    trj = torch.zeros(num_trjs, trj_len * sample_freq, dim).to(device)
    x = torch.zeros(num_trjs, dim).to(device)
    for it in range(trj_len * sample_freq):
        DriftForce = x @ Drift          # mechanical force
        DriftForce +=  - T * T0 * x * torch.exp(-0.5 * x**2)        # spurious force
        
        RanForce = torch.randn(num_trjs, dim, device=device)
        rfc = T * (1 + T0 * torch.exp(-0.5 * x**2))
        RanForce *= torch.sqrt(2*rfc)
        
        x += (DriftForce * dt + RanForce * dt2)
        
        trj[:, it, :] = x

    return trj[:, ::sample_freq]


def diff_mat(trajs, opt, device='cpu'):
    n_trj, trj_len, dim = trajs.shape
    
    Diffusion = torch.zeros(n_trj, trj_len, dim, dim)
    
    Diffusion[..., range(dim), range(dim)] = opt.Tc * (1 + T0 * torch.exp(-0.5 * trajs**2))
        
    return Diffusion


def drift_force(trajs, opt, device='cpu'):    
    n_trj, trj_len, dim = trajs.shape
        
    Drift = torch.zeros(dim, dim).to(device)  
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < dim - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    DriftForce = trajs @ Drift
    DriftForce += - opt.Tc * T0 * trajs * torch.exp(-0.5 * trajs**2)        # spurious force
    return DriftForce


def diff_mat_forSFI(trajs, Tc=5):
    trj_len, dim = trajs.shape

    # diff_shape = list(trajs.size())
    # diff_shape += [dim]
    
    Diffusion = np.zeros((trj_len, dim, dim))
    
    Diffusion[..., range(dim), range(dim)] = Tc * (1 + T0 * np.exp(-0.5 * trajs**2))
        
    return Diffusion


def drift_force_forSFI(trajs, Tc=5):    
    dim = trajs.shape[-1]
        
    Drift = np.zeros((dim, dim))
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < dim - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    DriftForce = trajs @ Drift
    DriftForce += - Tc * T0 * trajs * np.exp(-0.5 * trajs**2)        # spurious force
    return DriftForce



def spurious_force(trajs, opt, device='cpu'):            
    return - opt.Tc * T0 * trajs * torch.exp(-0.5 * trajs**2) 


def mechanical_force(trajs, opt, device='cpu'):    
    n_trj, trj_len, dim = trajs.shape
        
    Drift = torch.zeros(dim, dim).to(device)  
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < dim - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    DriftForce = trajs @ Drift
    return DriftForce

# def simulation_inhomogeneous(num_trjs, trj_len, dim, T, dt, sample_freq=10, seed=0, device='cpu'):
#     # reset the seed for generating trajectories
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     dt2 = np.sqrt(dt)
#     T_coeff = torch.ones(1, dim) * T

#     Drift = torch.zeros(dim, dim).to(device)    
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k / e
#         if i < dim - 1:
#             Drift[i][i + 1] = k / e
#         Drift[i][i] = -2 * k / e
    
#     trj = torch.zeros(num_trjs, trj_len * sample_freq, dim).to(device)
#     x = torch.zeros(num_trjs, dim).to(device)
#     for it in range(trj_len * sample_freq):
#         DriftForce = x @ Drift          # mechanical force
#         DriftForce += (2 * torch.ones(x.shape) * (x > 0) - 1) * T_coeff         # spurious force
        
#         RanForce = torch.randn(num_trjs, dim, device=device)
#         rfc = T0 + x.abs() * T_coeff
#         RanForce *= torch.sqrt(2*rfc)
        
#         x += (DriftForce * dt + RanForce * dt2)
        
#         trj[:, it, :] = x

#     return trj[:, ::sample_freq]


# def diff_mat(trajs, opt, device='cpu'):
#     n_trj, trj_len, dim = trajs.shape
    
#     Diffusion = torch.zeros(n_trj, trj_len, dim, dim)
#     for i in range(dim):
#         Diffusion[:, :, i, i] = T0 + trajs[:, :, i].abs() * opt.Tc
        
#     return Diffusion.to(opt.device)



# def drift_force(trajs, opt, device='cpu'):    
#     n_trj, trj_len, dim = trajs.shape
    
#     T_coeff = torch.ones(1, dim) * opt.Tc
    
#     Drift = torch.zeros(dim, dim).to(device)  
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k / e
#         if i < dim - 1:
#             Drift[i][i + 1] = k / e
#         Drift[i][i] = -2 * k / e
    
#     DriftForce = trajs @ Drift
#     DriftForce += (2 * torch.ones(trajs.shape) * (trajs > 0) - 1) * T_coeff
#     return DriftForce


# def mechanical_force(trajs, opt, device='cpu'):    
#     n_trj, trj_len, dim = trajs.shape
        
#     Drift = torch.zeros(dim, dim).to(device)  
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k / e
#         if i < dim - 1:
#             Drift[i][i + 1] = k / e
#         Drift[i][i] = -2 * k / e
    
#     DriftForce = trajs @ Drift
#     return DriftForce
