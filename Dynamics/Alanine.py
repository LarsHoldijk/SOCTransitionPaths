import os

import numpy as np
import torch

from Dynamics.MoleculeBase import MoleculeBaseDynamics
from potentials.alanine_md import AlaninePotentialMD


class AlanineDynamics(MoleculeBaseDynamics):
    def __init__(self, loss_func, n_samples=10, device='cpu', bridge=False, save_file=None):
        super().__init__(loss_func, n_samples, device, bridge, save_file)

    def _init_ending_positions(self):
        if self.bridge:
            n = 128
            path = './potentials/files/target_ax.npy'
        else:
            n = 1
            path = './potentials/files/target_ax_1.npy'

        if os.path.exists(path):
            print("Existing target points exits, loading them")
            positions = np.load(path)
            ending_positions = torch.as_tensor(positions)

        else:
            print("Generating target points")
            positions = []
            pot = AlaninePotentialMD('./potentials/files/AD_c7ax.pdb', -1)
            pot.simulation.minimizeEnergy()
            pot.simulation.step(1)
            for i in range(n):
                print(f"{i} of {n}")
                if i > 0:
                    pot.simulation.step(500)
                end_positions = torch.tensor(pot.reporter.latest_positions)
                positions.append(end_positions.clone())

            ending_positions = torch.stack(positions)
            np.save(path, ending_positions.detach().cpu().numpy())

        return ending_positions

    def _init_potentials(self):
        # Initialize potentials
        potentials = []
        for i in range(self.n_samples):
            pot = AlaninePotentialMD('./potentials/files/AD_c7eq.pdb', i, bridge=self.bridge, save_file=self.save_file)
            potentials.append(pot)

        return potentials
