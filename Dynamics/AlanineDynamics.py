import torch

from Dynamics.MoleculeBase import MoleculeBaseDynamics
from potentials.alanine_md import AlaninePotentialMD


class AlanineDynamics(MoleculeBaseDynamics):
    def __init__(self, loss_func, n_samples=10, device='cpu', save_file=None):
        super().__init__(loss_func, n_samples, device, save_file)

    def _init_ending_positions(self):
        print("Generating target points")
        pot = AlaninePotentialMD('./potentials/files/AD_c7ax.pdb', -1)
        pot.simulation.minimizeEnergy()
        pot.simulation.step(1)
        ending_positions = torch.tensor(pot.reporter.latest_positions)
        ending_positions = ending_positions.unsqueeze(dim=0)

        return ending_positions

    def _init_potentials(self):
        # Initialize potentials
        potentials = []
        for i in range(self.n_samples):
            pot = AlaninePotentialMD('./potentials/files/AD_c7eq.pdb', i, save_file=self.save_file)
            potentials.append(pot)

        return potentials

    def phi(self, x):
        end_ = self.ending_positions.view(self.ending_positions.shape[0], -1)
        x_ = x[:, :int(x.shape[1] / 2)]

        if self.loss_func == 'pairwise_dist':
            x_ = x_.view(x_.shape[0], -1, 3)
            end_ = end_.view(end_.shape[0], -1, 3)
            px = torch.cdist(x_, x_).unsqueeze(0)
            pend = torch.cdist(end_, end_).unsqueeze(1).repeat(1,
                                                               self.n_samples,
                                                               1, 1)

            t = (px - pend) ** 2
            cost_distance = torch.mean(t, dim=(2, 3))

            cost_distance_final = (cost_distance).exp() * 10 

            expected_cost_distance_final = torch.mean(cost_distance_final, 0)

        return expected_cost_distance_final
    
