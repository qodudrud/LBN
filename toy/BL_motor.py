import torch
import numpy as np
import scipy

# the Buttiker and Landauer motor (overdamped)
def simulation_BL_motor(num_trjs, trj_len, dim, T, dt, sample_freq=10, seed=0, device='cpu'):
    """Simulation of a bead-spring 2d-lattice model
    
    Args:
        num_beads : Number of beads for each row
        T1 : LeftUp-most temperature
        T2 : RighDown-tmost temperature
        dt : time step 
        trj_len : length of trajectories
        seed : seed of random generator
        num_trjs : Number of trajectories you want. default = 1000.

    Returns:
        trajectories of a bead-spring 2d-lattice model
    """ 
    L, k, T0 = 5, 2, 2
    
    # reset the seed for generating trajectories
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dt2 = np.sqrt(dt)
    
    trj_x = torch.zeros(num_trjs, trj_len * sample_freq, dim).to(device)
    trj_q = torch.zeros(num_trjs, trj_len * sample_freq, 2 * dim).to(device)
    
    x = L * torch.rand(dim, num_trjs).to(device)
    q = embedding(x, L = L)
    for it in range(trj_len * sample_freq):
        q = embedding(x, L = L)
        
        cosx = 2 * torch.pi * q[:dim, :]/L
        sinx = 2 * torch.pi * q[dim:, :]/L
        
        F0 = - k * (2 * torch.ones(x.shape) * ((x%L) <= L/2) - 1) + 2 * torch.pi * T * cosx/L
#         DriftForce = torch.cat((-F0 * sinx, F0 * cosx), dim=0)
        
        eta = torch.randn(dim, num_trjs, device=device)
        T_array = T0 + T * sinx
        sqrt_diffusion = torch.sqrt(2*T_array) * eta
        
        x += (F0 * dt + sqrt_diffusion * dt2)
        q = embedding(x, L = L)
        
        trj_x[:, it, :] = x.T
        trj_q[:, it, :] = q.T

    return trj_x[:, ::sample_freq], trj_q[:, ::sample_freq]


def embedding(x, L=5):
    cosx = torch.cos(2*torch.pi*x/L)
    sinx = torch.sin(2*torch.pi*x/L)
    return L * torch.cat((cosx, sinx), dim=0)/(2 * torch.pi)


def diff_mat(traj, T, T0=2, L=5, device='cpu'):
    n_trj, trj_len, dim = traj.shape
    
    cosx = torch.cos(2*torch.pi*traj/L)
    sinx = torch.cos(2*torch.pi*traj/L)
    
    T_tensor = T0 + T * sinx
    Diffusion = torch.zeros(n_trj, trj_len, dim, dim).float()
    for i in range(dim):
        Diffusion[:, :, i, i] = T_tensor[:, :, i]
    return Diffusion * torch.cat((-sinx, cosx), dim=-1).to(device)


def drift_force(traj, T, T0=2, L=5, k=2, device='cpu'):    
    cosx = torch.cos(2*torch.pi*traj/L)
    sinx = torch.cos(2*torch.pi*traj/L)
    
    F0 = - k * (2 * torch.ones(traj.shape) * ((traj%L) <= L/3) - 1) + 2 * torch.pi * T * cosx/L
    return F0 * torch.cat((-sinx, cosx), dim=-1).to(device)