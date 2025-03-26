import torch
import numpy as np
import scipy

def simulation_BrownianCarnotEngine(num_trjs, trj_len, dt, pre_cycle=100, sample_freq=10, seed=0, device='cpu'):
    e, k_M, T0 = 10, 40, 10
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dt2 = np.sqrt(dt)
        
    trj = torch.zeros(num_trjs, int(trj_len * sample_freq), 2).to(device)
    
    x = torch.zeros(1, num_trjs).to(device)
    v = torch.zeros(1, num_trjs).to(device)
    
    for it in range(trj_len):                
        DriftForce = - (k_M/9) * x - e * v

        RanForce = torch.randn(1, num_trjs, device=device)
        RanForce = np.sqrt(2 * T0 * e) * RanForce
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2
    
    for i in range(pre_cycle):
        for it in range(trj_len * sample_freq):
            k, T = update_param(it, k_M, T0, trj_len * sample_freq, dt)

            DriftForce = - k * x - e * v

            RanForce = torch.randn(1, num_trjs, device=device)
            RanForce = np.sqrt(2 * T * e) * RanForce

            x += v * dt
            v += DriftForce * dt + RanForce * dt2
    
    for it in range(trj_len * sample_freq):
        k, T = update_param(it, k_M, T0, trj_len * sample_freq, dt)
                
        DriftForce = - k * x - e * v

        RanForce = torch.randn(1, num_trjs, device=device)
        RanForce = np.sqrt(2 * T * e) * RanForce
        
        x += v * dt
        v += DriftForce * dt + RanForce * dt2

        trj[:, it, :] = torch.cat((x, v), axis=0).T

    return trj[:, ::sample_freq]


def update_param(it, k_max, T0, tot_trj_len, dt):
    k0 = k_max/9
    slope_k = 32 * k_max/(9 * (tot_trj_len * dt)**2)
    
    k_half = k0 + slope_k * ((tot_trj_len / 2) * dt)**2
    T_half = T0 * np.sqrt(k_half/(k0 + slope_k * (tot_trj_len * dt / 4)**2))
    
    k, T = None, None
    if it <= tot_trj_len/4:
        k = k0 + slope_k * (it * dt)**2
        T = T0
        
    elif it <= 2 * tot_trj_len/4:
        k = k0 + slope_k * (it * dt)**2
        T = T0 * np.sqrt(k/(k0 + slope_k * (tot_trj_len * dt / 4)**2))
        
    elif it <= 3 * tot_trj_len/4:
        k = k0 + slope_k * ((tot_trj_len - it) * dt)**2
        T = T_half
        
    else:
        k = k0 + slope_k * ((tot_trj_len - it) * dt)**2
        T = T_half * np.sqrt(k/(k0 + slope_k * ((tot_trj_len/4) * dt)**2))
        
    return k, T


def diff_mat(T_list, device='cpu'):
    e = 10
    return T_list * e


def drift_force(traj, k_list, device='cpu'):
    e = 10.
    traj_x, traj_v = traj[..., 0], traj[..., 1]
    return -k_list * traj_x - e * traj_v


def ret_protocols(tot_trj_len, dt, k_M=40, T0=10):
    k_list, T_list = [], []
    for i in range(tot_trj_len):
        k, T = update_param(i, k_M, T0, tot_trj_len, dt)
        k_list.append(k)
        T_list.append(T)
        
    return torch.tensor(k_list), torch.tensor(T_list)
