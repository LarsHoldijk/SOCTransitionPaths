import os
from abc import abstractmethod, ABC

import numpy as np
import torch
from einops import einops

from potentials.alanine_md import AlaninePotentialMD


class MoleculeBaseDynamics(ABC):
    def __init__(self, loss_func, n_samples=10, device='cpu', save_file=None):
        self.n_samples = n_samples
        self.loss_func = loss_func
        self.device = device
        self.save_file = save_file

        self.ending_positions = self._init_ending_positions()
        self.ending_positions = self.ending_positions.to(self.device)

        self.potentials = self._init_potentials()

        self.dims = self.ending_positions.shape[1] * self.ending_positions.shape[2]

        top = torch.zeros(self.dims, self.dims)
        bot = torch.eye(self.dims)  

        self.G_matrix = torch.vstack([top, bot])
        self.G_matrix = einops.repeat(self.G_matrix, 'm n -> k m n', k=n_samples).to(device)


    @abstractmethod
    def _init_potentials(self):
        pass

    @abstractmethod
    def _init_ending_positions(self):
        pass

    def f(self, x, t):
        pos = x[:, :int(x.shape[1] / 2)].view(self.n_samples, -1, 3)
        vel = x[:, int(x.shape[1] / 2):].view(self.n_samples, -1, 3)
        vel_np = vel.detach().cpu().numpy()

        ps = []
        for i in range(self.n_samples):
            _p, _v = self.potentials[i].drift(vel_np[i, :, :])
            ps.append(_p)

        ps = torch.tensor(np.array(ps), dtype=torch.float, device=self.device)
        dx = ps - pos
        dx_ = dx.view(self.n_samples, -1)
        dv_ = -vel.view(self.n_samples, -1)

        comb = torch.cat([dx_, dv_], dim=1)

        return comb

    def G(self, x):
        return self.G_matrix


    def q(self, x):
        return 0.

    def starting_positions(self, n_samples):
        initial_positions = []
        for i in range(self.n_samples):
            initial_positions.append(
                torch.tensor(self.potentials[i].reporter.latest_positions))
        initial_positions = torch.stack(initial_positions).to(self.device)
        initial_positions = initial_positions.view(n_samples, -1)

        # Add static velocities
        init_velocities = torch.zeros_like(initial_positions)
        stacked = torch.cat([initial_positions, init_velocities], dim=1)

        return stacked

    def reset(self):
        for i in range(self.n_samples):
            self.potentials[i].reset()