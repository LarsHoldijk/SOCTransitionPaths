import os
import time

import einops as einops
import numpy as np
import torch
import torch.nn as nn

from Dynamics.PolytrimerDynamics import PolyDynamics
from plotting.Loggers import CostsLogger
from solvers.PICE import PICE
from policies.Poly import NNPolicy

# seed
torch.manual_seed(42)

# File setup
file = "./results/Polyproline"

force = False # if false -> use energy

device = 'cuda'

T = 5000.
dt = torch.tensor(1.)
n_steps = int(T / dt)

n_rollouts = 10000
n_samples = 16

lr = 0.0001


environment = PolyDynamics(loss_func='pairwise_dist', n_samples=n_samples, device=device, save_file=file)

dims = environment.dims

std = torch.tensor(.05).to(device)
R = torch.eye(dims).to(device)

logger = CostsLogger(f'{file}')

nn_policy = NNPolicy(device, dims = dims, force=force, T=T)

PICE(environment, nn_policy, n_rollouts, n_samples, n_steps, dt, std, dims * 2, R, logger, force, [], True, file, device=device, lr=lr)

torch.save(nn_policy, f'{file}/final_policy')

# save trajectory to pdbs
name = 'force' if force else 'energy'
new_traj = np.load(f'{file}/phi_paths.npy')
new_traj = new_traj[:, :, :int(new_traj.shape[2]/2)]

STEP=25 # draw every STEP steps
N=16 # num of trajectories

for i in range(N):
    trajs = None
    for j in range(0, int(new_traj.shape[1]), STEP):
        traj = md.load_pdb('./potentials/files/3mer_pp1.pdb')
        atoms = []
        for index_atom in range(0, 138):
            atom_location = new_traj[i, j, index_atom*3:index_atom*3+3]
            atoms.append(atom_location)
        atoms = np.array(atoms)
        traj.xyz = np.array(atoms)
        if j == 0:
            trajs = traj
        else:
            trajs = trajs.join(traj)
    trajs.save(f'./save_pdbs/save_{i}.pdb')