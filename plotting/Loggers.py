import numpy as np
import torch


class CostsLogger():
    def __init__(self, file):
        super().__init__()
        self.file = file
        self.min_path_cost = np.inf
        self.min_phi_cost = np.inf


    def save(self, fix, paths, costs_q, costs_noise, costs_action, policy):
        torch.save(policy, f'{self.file}/{fix}_policy')
        np.save(f'{self.file}/{fix}_paths.npy', paths.detach().cpu().numpy())
        np.save(f"{self.file}/{fix}_q.npy", costs_q.detach().cpu().numpy())
        np.save(f"{self.file}/{fix}_noise.npy", costs_noise.detach().cpu().numpy())
        np.save(f"{self.file}/{fix}_action.npy", costs_action.detach().cpu().numpy())


    def log(self, paths, costs_q, costs_noise, costs_action, path_cost, path_cost_phi, path_cost_final, path_cost_exp_normalized, policy, step):
        path_cost_final_current = path_cost_final.mean(dim=0)
        path_cost_phi_current = path_cost_phi.mean(dim=0)

        print(f"[{step}] -- mean phi: {path_cost_phi_current}, mean final cost: {path_cost_final_current}")

        if self.min_path_cost > path_cost_final_current:
            print("Final cost improved, saving")
            self.min_path_cost = path_cost_final_current
            self.save("path", paths, costs_q, costs_noise, costs_action, policy)

        if self.min_phi_cost > path_cost_phi_current:
            print("Phi cost improved, saving")
            self.min_phi_cost = path_cost_phi_current
            self.save("phi", paths, costs_q, costs_noise, costs_action, policy)

        line = str(step)
        line += f', {costs_q.sum(dim=1).mean(dim=0)}, {costs_q.sum(dim=1).std(dim=0)}'
        line += f', {costs_noise.sum(dim=1).mean(dim=0)}, {costs_noise.sum(dim=1).std(dim=0)}'
        line += f', {costs_action.sum(dim=1).mean(dim=0)}, {costs_action.sum(dim=1).std(dim=0)}'
        line += f', {path_cost.mean(dim=0)}, {path_cost.std(dim=0)}'
        line += f', {path_cost_phi_current}, {path_cost_phi.std(dim=0)}'
        line += f', {path_cost_final_current}, {path_cost_final.std(dim=0)}'
        line += f', {path_cost_exp_normalized.mean(dim=0)}, {path_cost_exp_normalized.std(dim=0)} \n'

        with open(f"{self.file}/costs.txt", 'a+') as f:
            f.write(line)
