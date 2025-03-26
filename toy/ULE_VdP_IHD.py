import torch
import numpy as np

# Van der Pol oscilator model with inertia
def simulation_ULE_VdP(num_trjs, trj_len, dim, T, dt, sample_freq=10, seed=0, device='cpu'):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    k = 2
    
    dt2 = np.sqrt(dt)
    
    trj = torch.zeros(num_trjs, trj_len * sample_freq, 2*dim).to(device)
    
    x = torch.zeros(dim, num_trjs).to(device)
    v = torch.zeros(dim, num_trjs).to(device)
    
    for _ in range(10000):
        DriftForce = k * ( 1 - x**2) * v - x

        RanForce = torch.randn(dim, num_trjs, device=device)
        rfc = T * (1 + 0.3 *x**2 + 0.1 * v**2)
        # rfc = T * (1 + 2 * np.exp(-0.5 * x**2) - np.exp(-0.2 * v**2))
        RanForce *= torch.sqrt(2*rfc)
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2
    
    for it in range(trj_len * sample_freq):
        DriftForce = k * ( 1 - x**2) * v - x

        RanForce = torch.randn(dim, num_trjs, device=device)
        rfc = T * (1 + 0.3 *x**2 + 0.1 * v**2)
        # rfc = T * (1 + 2 * np.exp(-0.5 * x**2) - np.exp(-0.2 * v**2))
        RanForce *= torch.sqrt(2*rfc)
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2

        trj[:, it, :] = torch.cat((x, v), axis=0).T

    return trj[:, ::sample_freq]


def diff_mat(trajs, opt):
    n_trj, trj_len, dim = trajs.shape

    pos_trajs, vel_trajs = trajs[:, :, :dim//2], trajs[:, :, dim//2:]
    
    Diffusion = torch.zeros(n_trj, trj_len, dim//2, dim//2)
    
    Diffusion[..., range(dim//2), range(dim//2)] = opt.Tc * (1 + 0.3 *pos_trajs**2 + 0.1 * vel_trajs**2)
    # Diffusion[..., range(dim//2), range(dim//2)] = opt.Tc * (1 + 2 * np.exp(-0.5 * pos_trajs**2) - np.exp(-0.2 * vel_trajs**2))
        
    return Diffusion


def drift_force(trajs, opt):
    n_trj, trj_len, dim = trajs.shape
    k = 2

    pos_trajs, vel_trajs = trajs[:, :, :dim//2], trajs[:, :, dim//2:]
    return k * ( 1 - pos_trajs**2) * vel_trajs - pos_trajs



def drift_force_forSFI(pos_trajs, vel_trajs):
    k = 2

    return k * ( 1 - pos_trajs**2) * vel_trajs - pos_trajs


def diff_mat_forSFI(pos_trajs, vel_trajs, Tc = 1):
    trj_len, dim = pos_trajs.shape

    Diffusion = np.zeros((trj_len, dim, dim))
    
    Diffusion[..., range(dim), range(dim)] = Tc * (1 + 0.3 *pos_trajs**2 + 0.1 * vel_trajs**2)
    # Diffusion[..., range(dim//2), range(dim//2)] = Tc * (1 + 2 * np.exp(-0.5 * pos_trajs**2) - np.exp(-0.2 * vel_trajs**2))

    return Diffusion