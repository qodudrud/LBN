import torch
import numpy as np
import scipy

alpha_n = lambda V: 0.01 * (10 - V)/(torch.exp((10-V)/10)-1)
beta_n = lambda V: 0.125 * torch.exp(-V/80)
alpha_m = lambda V: 0.1 * (25 - V)/(torch.exp((25-V)/10)-1)
beta_m = lambda V: 4 * torch.exp(-V/18)
alpha_h = lambda V: 0.07 * torch.exp(-V/20)
beta_h = lambda V: 1/(torch.exp((30-V)/10)+1)

def ret_alphas(V):
    return torch.stack([alpha_n(V), alpha_m(V), alpha_h(V)], 0)

def ret_betas(V):
    return torch.stack([beta_n(V), beta_m(V), beta_h(V)], 0)


def simulation_HHmodel(num_trjs, trj_len, T, dt, I_ext=0, sample_freq=10, device='cpu', sys_seed=0, seed=0):
    dt2 = np.sqrt(dt)
    dim = 4
    
    C, g_Na, g_K, g_L = 1, 120, 36, 0.3
    E_Na, E_K, E_L = 115, -12, 10.6

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
    
    # initialize. x[0] = V, x[1] = n, x[2] = m, x[3] = h
    # x[0] ~ Uinform from -20 to 20
    x = 40 * (torch.rand(dim, num_trjs).to(device) - 0.5)

    alphas, betas = ret_alphas(x[0]), ret_betas(x[0])
    x[1:] = alphas/(alphas+betas)
    for it in range(trj_len * sample_freq):
        alphas, betas = ret_alphas(x[0]), ret_betas(x[0])
        
        DriftForce = torch.zeros(dim, num_trjs)
        
        DriftForce[0] = ((-(g_Na * x[2]**3 * x[3] * (x[0]-E_Na) + 
                     g_K * x[1]**4 * (x[0]-E_K) +
                     g_L * (x[0]-E_L)) + I_ext)/C)
        DriftForce[1:] = alphas * (1-x[1:]) - betas * x[1:]

        RanForce = torch.randn(dim, num_trjs, device=device)
        RanForce = sqrt_diffusion @ RanForce

        x += (DriftForce * dt + RanForce * dt2)

        # to avoid nan
        x[0][x[0] == 10] += 1e-6
        x[0][x[0] == 25] += 1e-6
        x[0][x[0] == 30] += 1e-6

        # n, m, h should be in [0, 1]
        x[1:] = torch.clip(x[1:], 0, 1)

        trj[:, it, :] = x.T

    return trj[:, ::sample_freq]


def diff_mat(trajs, opt, sys_seed=0, device='cpu'):
    np.random.seed(sys_seed)
    torch.manual_seed(sys_seed)

    diffusion_mat = torch.zeros(opt.dim, opt.dim).to(device)
    # diffusion_mat[0, 0] = 1
    diffusion_mat[1, 1] = opt.Tc
    diffusion_mat[2, 2] = opt.Tc
    diffusion_mat[3, 3] = opt.Tc

#     flag = True
#     while flag:
#         # generate diffusion matrix using the Cholesky decomposition
#         diag_ind = torch.stack([torch.arange(opt.dim), torch.arange(opt.dim)])

#         scale_tril = torch.rand(opt.dim, opt.dim).to(device)
#         scale_tril = (2 * scale_tril - 1) * np.sqrt(opt.Tc)
#         scale_tril[diag_ind[0], diag_ind[1]] = torch.abs(scale_tril[diag_ind[0], diag_ind[1]])

#         scale_tril = torch.tril(scale_tril)
#         diffusion_mat = scale_tril @ scale_tril.T
        
# #         flag = torch.any(torch.abs(diffusion_mat) < 0.0006)
#         flag = False
    
    return diffusion_mat


def drift_force(trajs, opt, I_ext=0, device='cpu'):
    C, g_Na, g_K, g_L = 1, 120, 36, 0.3
    E_Na, E_K, E_L = 115, -12, 10.6

    DriftForce = torch.zeros_like(trajs)

    DriftForce[..., 0] = ((-(g_Na * trajs[..., 2]**3 * trajs[..., 3] * (trajs[..., 0]-E_Na) + 
                 g_K * trajs[..., 1]**4 * (trajs[..., 0]-E_K) +
                 g_L * (trajs[..., 0]-E_L)) + I_ext)/C)
    DriftForce[..., 1] = alpha_n(trajs[..., 0]) * (1-trajs[..., 1]) - beta_n(trajs[..., 0]) * trajs[..., 1]
    DriftForce[..., 2] = alpha_m(trajs[..., 0]) * (1-trajs[..., 2]) - beta_m(trajs[..., 0]) * trajs[..., 2]
    DriftForce[..., 3] = alpha_h(trajs[..., 0]) * (1-trajs[..., 3]) - beta_h(trajs[..., 0]) * trajs[..., 3]

    return DriftForce

