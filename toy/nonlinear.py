import torch
import numpy as np
import scipy.linalg

k = 2


def simulation_nonlinear(num_trjs, trj_len, dim, T, dt, sample_freq=10, device='cpu', sys_seed=0, seed=0, alpha=10):
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
    # reset the seed for generating constants of system, i.e., diffusion tensor.
    dt2 = np.sqrt(dt)
    
    Drift = torch.zeros(dim, dim).to(device)
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k
        if i < dim - 1:
            Drift[i][i + 1] = -k
        Drift[i][i] = -k
    
    from argparse import Namespace
    opt = Namespace()
    opt.Tc = T
    opt.dim = dim

    diffusion_mat = diff_mat(None, opt, sys_seed=sys_seed, device=device)
    sqrt_diffusion = np.sqrt(2) * torch.FloatTensor(scipy.linalg.sqrtm(diffusion_mat))
    
    # reset the seed for generating trajectories
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    trj = torch.zeros(num_trjs, trj_len * sample_freq, dim).to(device)
    
    x = torch.zeros(dim, num_trjs).to(device)
    for it in range(trj_len * sample_freq):
        DriftForce = Drift @ x
        DriftForce += alpha * torch.exp(-x**2 / 2) * x
        
        RanForce = torch.randn(dim, num_trjs, device=device)
        RanForce = sqrt_diffusion @ RanForce

        x += (DriftForce * dt + RanForce * dt2)

        trj[:, it, :] = x.T

    return trj[:, ::sample_freq]


# def diff_mat(trajs, opt, sys_seed=0, device='cpu'):
#     np.random.seed(sys_seed)
#     torch.manual_seed(sys_seed)

#     flag = True
#     while flag:
#         # generate diffusion matrix using the Cholesky decomposition
#         diag_ind = torch.stack([torch.arange(opt.dim), torch.arange(opt.dim)])

#         scale_tril = torch.rand(opt.dim, opt.dim).to(device)
#         scale_tril = (2 * scale_tril - 1) * np.sqrt(opt.Tc)
#         scale_tril[diag_ind[0], diag_ind[1]] = torch.abs(scale_tril[diag_ind[0], diag_ind[1]])

#         scale_tril = torch.tril(scale_tril)
#         diffusion_mat = scale_tril @ scale_tril.T
        
#         flag = torch.any(torch.abs(diffusion_mat) < 0.06)
    
#     return diffusion_mat

def diff_mat(trajs, opt, sys_seed=0, device='cpu'):
    diffusion_mat = torch.zeros(opt.dim, opt.dim)
    for i in range(opt.dim):
        if i > 0:
            diffusion_mat[i][i - 1] = -opt.Tc**(1/2)
        if i < opt.dim - 1:
            diffusion_mat[i][i + 1] = -opt.Tc**(1/2)
        diffusion_mat[i][i] = opt.Tc
    
    return diffusion_mat
    

def drift_force(trajs, opt, device='cpu'):
    dim = trajs.shape[-1]

    alpha = opt.alpha
    
    Drift = torch.zeros(dim, dim).to(device)
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k
        if i < dim - 1:
            Drift[i][i + 1] = -k
        Drift[i][i] = -k

    DriftForce = trajs @ Drift.T
    DriftForce += alpha * torch.exp(-trajs**2 / 2) * trajs

    return DriftForce


def drift_force_forSFI(trajs):
    alpha = 10

    dim = trajs.shape[-1]
    
    Drift = np.zeros((dim, dim))
    for i in range(dim):
        if i > 0:
            Drift[i][i - 1] = k
        if i < dim - 1:
            Drift[i][i + 1] = -k
        Drift[i][i] = -k

    DriftForce = trajs @ Drift.T
    DriftForce += alpha * np.exp(-trajs**2 / 2) * trajs

    return DriftForce


def diff_mat_forSFI(trajs, opt, sys_seed=0):
    return diff_mat(trajs, opt, sys_seed=sys_seed).numpy()



# def simulation_nonlinear(num_trjs, trj_len, dim, T, dt, sample_freq=10, device='cpu', sys_seed=0, seed=0, alpha=10):
#     """Simulation of a bead-spring 2d-lattice model
    
#     Args:
#         num_beads : Number of beads for each row
#         T1 : LeftUp-most temperature
#         T2 : RighDown-tmost temperature
#         dt : time step 
#         trj_len : length of trajectories
#         seed : seed of random generator
#         num_trjs : Number of trajectories you want. default = 1000.

#     Returns:
#         trajectories of a bead-spring 2d-lattice model
#     """    
#     # reset the seed for generating constants of system, i.e., diffusion tensor.
#     dt2 = np.sqrt(dt)
    
#     Drift = torch.zeros(dim, dim).to(device)
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k
#         if i < dim - 1:
#             Drift[i][i + 1] = -k
#         Drift[i][i] = -k
    
#     from argparse import Namespace
#     opt = Namespace()
#     opt.Tc = T
#     opt.dim = dim

#     diffusion_mat = diff_mat(None, opt, sys_seed=sys_seed, device=device)
#     sqrt_diffusion = np.sqrt(2) * torch.FloatTensor(scipy.linalg.sqrtm(diffusion_mat))
    
#     # reset the seed for generating trajectories
#     np.random.seed(seed)
#     torch.manual_seed(seed)
    
#     trj = torch.zeros(num_trjs, trj_len * sample_freq, dim).to(device)
    
#     x = torch.zeros(dim, num_trjs).to(device)
#     for it in range(trj_len * sample_freq):
#         DriftForce = Drift @ x
#         DriftForce += alpha * torch.exp(-(x**2).sum(axis=0) / 2).unsqueeze(0) * x
        
#         RanForce = torch.randn(dim, num_trjs, device=device)
#         RanForce = sqrt_diffusion @ RanForce

#         x += (DriftForce * dt + RanForce * dt2)

#         trj[:, it, :] = x.T

#     return trj[:, ::sample_freq]


# def diff_mat(trajs, opt, sys_seed=0, device='cpu'):
#     np.random.seed(sys_seed)
#     torch.manual_seed(sys_seed)

#     flag = True
#     while flag:
#         # generate diffusion matrix using the Cholesky decomposition
#         diag_ind = torch.stack([torch.arange(opt.dim), torch.arange(opt.dim)])

#         scale_tril = torch.rand(opt.dim, opt.dim).to(device)
#         scale_tril = (2 * scale_tril - 1) * np.sqrt(opt.Tc)
#         scale_tril[diag_ind[0], diag_ind[1]] = torch.abs(scale_tril[diag_ind[0], diag_ind[1]])

#         scale_tril = torch.tril(scale_tril)
#         diffusion_mat = scale_tril @ scale_tril.T
        
#         flag = torch.any(torch.abs(diffusion_mat) < 0.06)
    
#     return diffusion_mat


# def drift_force(trajs, opt, device='cpu'):
#     dim = trajs.shape[-1]

#     alpha = opt.alpha
    
#     Drift = torch.zeros(dim, dim).to(device)
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k
#         if i < dim - 1:
#             Drift[i][i + 1] = -k
#         Drift[i][i] = -k

#     DriftForce = trajs @ Drift.T
#     DriftForce += alpha * torch.exp(-(trajs**2).sum(axis=-1) / 2).unsqueeze(-1) * trajs

#     return DriftForce


# def drift_force_forSFI(trajs):
#     alpha = 10

#     dim = trajs.shape[-1]
    
#     Drift = np.zeros((dim, dim))
#     for i in range(dim):
#         if i > 0:
#             Drift[i][i - 1] = k
#         if i < dim - 1:
#             Drift[i][i + 1] = -k
#         Drift[i][i] = -k

#     DriftForce = trajs @ Drift.T
#     DriftForce += alpha * np.expand_dims(np.exp(-(trajs**2).sum(axis=-1) / 2), axis=-1) * trajs

#     return DriftForce


# def diff_mat_forSFI(trajs, opt, sys_seed=0):
#     return diff_mat(trajs, opt, sys_seed=sys_seed).numpy()