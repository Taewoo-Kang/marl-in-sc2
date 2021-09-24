import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, state_shape, n_agents, hyper_embed_dim, mixing_embed_dim):
        super(HyperNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_shape = state_shape
        self.hyper_embed_dim = hyper_embed_dim
        self.mixing_embed_dim = mixing_embed_dim

        self.w1_layer = nn.Sequential(nn.Linear(self.state_shape, self.hyper_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_embed_dim, self.mixing_embed_dim * self.n_agents))
        self.b1_layer = nn.Linear(self.state_shape, self.mixing_embed_dim)
        self.w2_layer = nn.Sequential(nn.Linear(self.state_shape, self.hyper_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.hyper_embed_dim, self.mixing_embed_dim * 1))
        self.b2_layer = nn.Sequential(nn.Linear(self.state_shape, self.mixing_embed_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.mixing_embed_dim, 1))

        # self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w1_layer.weight)
        nn.init.xavier_uniform_(self.w2_layer.weight)
        nn.init.xavier_uniform_(self.b1_layer.weight)
        nn.init.xavier_uniform_(self.b2_layer.weight)

    def forward(self, state):
        w1 = torch.abs(self.w1_layer(state)).view(-1, self.n_agents, self.mixing_embed_dim)
        b1 = self.b1_layer(state).view(-1, 1, self.mixing_embed_dim)
        w2 = torch.abs(self.w2_layer(state)).view(-1, self.mixing_embed_dim, 1)
        b2 = self.b2_layer(state).view(-1, 1, 1)
        return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


class MixingNetwork(nn.Module):
    def __init__(self, batch_size):
        super(MixingNetwork, self).__init__()
        self.batch_size = batch_size

    def forward(self, q_values, hyper_params):
        hidden = F.elu(torch.bmm(q_values, hyper_params['w1']) + hyper_params['b1'])
        y = torch.bmm(hidden, hyper_params['w2']) + hyper_params['b2']
        q_tot = y.view(self.batch_size, -1, 1)
        return q_tot
