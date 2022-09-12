import torch

from Dynamics.PolytrimerDynamics import PolyDynamics
from plotting.Loggers import CostsLogger
from policies.Poly import NNPolicy
from solvers.PICE import PICE

# seed
torch.manual_seed(42)

# File setup
file = "./results/Polyproline"

force = False  # if false -> use energy

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

nn_policy = NNPolicy(device, dims=dims, force=force, T=T)

PICE(environment, nn_policy, n_rollouts, n_samples, n_steps, dt, std, dims * 2, R, logger, force, [], True, file,
     device=device, lr=lr)

torch.save(nn_policy, f'{file}/final_policy')
