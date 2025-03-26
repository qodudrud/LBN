import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

# Spring and Stokes friction coefficient
k, e = 2, 1

def simulation(num_trjs, trj_len, num_beads, T1, T2, dt, sample_freq=10, device='cpu', seed=0):
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
    torch.manual_seed(seed)

    T = torch.linspace(T1, T2, num_beads).to(device)
    dt2 = np.sqrt(dt)
    
    trj = torch.zeros(num_trjs, trj_len * sample_freq, num_beads).to(device)
    Drift = torch.zeros(num_beads, num_beads).to(device)
    position = torch.zeros(num_trjs, num_beads).to(device)
    
    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e

    rfc = torch.zeros(num_beads).to(device)
    for i in range(num_beads):
        rfc[i] = torch.sqrt(2 * e * T[i])
            
    for it in range(trj_len * sample_freq):
        RanForce = torch.randn(num_trjs, num_beads, device=device)
        RanForce *= rfc

        DriftForce = torch.einsum('ij,aj->ai', Drift, position)

        position += (DriftForce * dt + RanForce * dt2) / e

        trj[:, it, :] = position

    return trj[:, ::sample_freq]


def drift_force(trajs):
    num_beads = trajs.shape[-1]

    Drift = torch.zeros(num_beads, num_beads).to(trajs.device)

    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    return trajs @ Drift


def diff_mat(num_beads, T1, T2):
    T = torch.linspace(T1, T2, num_beads)
    Diffusion = torch.zeros(num_beads, num_beads)

    for i in range(num_beads):
        Diffusion[i, i] = T[i]
    
    return Diffusion


def drift_force_UDP(trajs, vel_trajs):
    num_beads = trajs.shape[-1]

    Drift = torch.zeros(num_beads, num_beads).to(trajs.device)

    for i in range(num_beads):
        if i > 0:
            Drift[i][i - 1] = k / e
        if i < num_beads - 1:
            Drift[i][i + 1] = k / e
        Drift[i][i] = -2 * k / e
    
    return -e * vel_trajs + trajs @ Drift