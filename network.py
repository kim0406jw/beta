import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Beta
from utils import weight_init


class Twin_Q_net(nn.Module):
    def __init__(self, state_dim, action_dim, device, hidden_dims=(256, 256), activation_fc=F.relu):
        super(Twin_Q_net, self).__init__()
        self.device = device

        self.activation_fc = activation_fc

        self.input_layer_A = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_A = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer_A = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers_A.append(hidden_layer_A)
        self.output_layer_A = nn.Linear(hidden_dims[-1], 1)

        self.input_layer_B = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.hidden_layers_B = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer_B = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers_B.append(hidden_layer_B)
        self.output_layer_B = nn.Linear(hidden_dims[-1], 1)
        self.apply(weight_init)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            u = u.unsqueeze(0)

        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = torch.cat([x, u], dim=1)

        x_A = self.activation_fc(self.input_layer_A(x))
        for i, hidden_layer_A in enumerate(self.hidden_layers_A):
            x_A = self.activation_fc(hidden_layer_A(x_A))
        x_A = self.output_layer_A(x_A)

        x_B = self.activation_fc(self.input_layer_B(x))
        for i, hidden_layer_B in enumerate(self.hidden_layers_B):
            x_B = self.activation_fc(hidden_layer_B(x_B))
        x_B = self.output_layer_B(x_B)

        return x_A, x_B


class BetaPolicy(nn.Module):
    def __init__(self, args, state_dim, action_dim, action_bound,
                 hidden_dims=(256, 256), activation_fc=F.relu, device='cuda'):
        super(BetaPolicy, self).__init__()
        self.device = device
        self.Beta_low_bound = args.log_Beta_bound[0]
        self.Beta_high_bound = args.log_Beta_bound[1]
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.alpha_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.beta_layer = nn.Linear(hidden_dims[-1], action_dim)

        self.action_rescale = action_bound[1] - action_bound[0]
        self.action_rescale_bias = (action_bound[0] - action_bound[1]) / 2
        self.action_rescale_t = torch.as_tensor(self.action_rescale, dtype=torch.float32)
        self.action_rescale_bias_t = torch.as_tensor(self.action_rescale_bias, dtype=torch.float32)

        self.apply(weight_init)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            x = self.activation_fc(hidden_layer(x))
        log_alpha = self.alpha_layer(x)
        log_beta = self.beta_layer(x)
        log_alpha = torch.clamp(log_alpha, self.Beta_low_bound, self.Beta_high_bound)
        log_beta = torch.clamp(log_beta, self.Beta_low_bound, self.Beta_high_bound)

        return log_alpha, log_beta

    def sample(self, state, eval=False):
        log_alpha, log_beta = self.forward(state)
        distribution = Beta(log_alpha.exp(), log_beta.exp())

        action = distribution.rsample()

        log_prob = distribution.log_prob(action)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        if eval is True:
            alpha = log_alpha.exp()[0].cpu().numpy()
            beta = log_beta.exp()[0].cpu().numpy()

            action = []
            for i in range(len(alpha)):
                if alpha[i] > 1 and beta[i] > 1:
                    a = (alpha[i] - 1) / ((alpha[i] + beta[i] - 2) + 1e-7)  # Calculate Mode.
                    action.append(a)
                else:
                    a = np.random.beta(alpha[i], beta[i])  # Sample if the Mode is not defined well.
                    action.append(a)
            action = np.array(action)
            system_action = action * self.action_rescale + self.action_rescale_bias
            return system_action
        else:
            system_action = action * self.action_rescale_t + self.action_rescale_bias_t
            return system_action, log_prob




