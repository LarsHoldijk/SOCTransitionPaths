import torch
from torch import nn


class NNPolicy(nn.Module):
    def __init__(self, device, force, dims, T):
        super().__init__()
        self.device = device
        self.force = force
        self.T = T
        self.dims = dims

        self.input_dims = self.dims

        if not self.force:
            self.output_dim = 1
            self.inc_last_bias = False
        else:
            self.output_dim = 22 * 3
            self.inc_last_bias = True


        self.linear_relu_stack = nn.Sequential(
                nn.Linear(self.input_dims, 128), # M x (3 x 2)
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim, bias=self.inc_last_bias)
            )

        self.to(self.device)

    def forward(self, x_in, t):
        x = x_in.clone()
        x = x[:, :int(x.shape[1] / 2)].view(x.shape[0], -1)

        if not self.force:
            x.requires_grad = True

        x_ = x

        out = self.linear_relu_stack(x_)
        if not self.force:
            out_grad = -torch.autograd.grad(outputs=out, inputs=x, grad_outputs=torch.ones_like(out), create_graph=True, retain_graph=True)[0]
            out_grad = out_grad.view(-1, 22 * 3) * 10
            return out, out_grad
        else:
            out = out.view(-1, 22 * 3)
            return out
