import torch.nn as nn
import torch.nn.functional as F


class AgentNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions):
        super(AgentNetwork, self).__init__()
        self.hidden_dim = hidden_dim

        self.mlp_in_layer = nn.Linear(input_shape, hidden_dim)
        self.gru_layer = nn.GRUCell(hidden_dim, hidden_dim)
        self.mlp_out_layer = nn.Linear(hidden_dim, n_actions)
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mlp_in_layer.weight)
        nn.init.xavier_uniform_(self.mlp_out_layer.weight)

    def init_hidden(self):
        return self.mlp_in_layer.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.mlp_in_layer(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h_out = self.gru_layer(x, h_in)
        q = self.mlp_out_layer(h_out)
        return q, h_out
