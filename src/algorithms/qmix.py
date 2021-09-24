import os
from copy import deepcopy
import torch
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from src.replay_buffer import ReplayBuffer
from src.modules.agent import AgentNetwork
from src.modules.qmixer import MixingNetwork, HyperNetwork


class QmixNetwork:
    def __init__(self, state_shape, obs_shape, n_actions, n_agents, args):
        self.state_shape = state_shape
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.input_shape = self.obs_shape + self.n_actions + self.n_agents

        self.agent_hidden_dim = args.agent_hidden_dim
        self.hyper_embed_dim = args.hyper_embed_dim
        self.mixing_embed_dim = args.mixing_embed_dim

        self.device = args.device
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.epsilon = 0 if args.test_mode else args.epsilon
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm

        self.memory = ReplayBuffer(args.memory_size)
        self._init_trainers()
        self.hidden_states = None

        self.target_update_interval = args.target_update_interval
        self.last_update_cnt = 0

    def _init_trainers(self):
        self.agent_net = AgentNetwork(self.input_shape, self.agent_hidden_dim, self.n_actions).to(self.device)
        self.hyper_net = HyperNetwork(self.state_shape, self.n_agents, self.hyper_embed_dim, self.mixing_embed_dim).to(self.device)
        self.mixing_net = MixingNetwork(self.batch_size).to(self.device)
        self.agent_net_tar = deepcopy(self.agent_net).to(self.device)
        self.hyper_net_tar = deepcopy(self.hyper_net).to(self.device)

        self.params = list(self.agent_net.parameters())
        self.params += list(self.hyper_net.parameters())
        self.optimizer = torch.optim.RMSprop(params=self.params, lr=self.lr)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent_net.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)

    def save_memory(self, inputs, state, actions, avail_actions_new, inputs_new, state_new, reward, done):
        reward = np.array([reward])[np.newaxis, :]
        done = np.array([done])[np.newaxis, :]
        self.memory.add((inputs[np.newaxis, :], state[np.newaxis, :], actions[np.newaxis, :],
                         avail_actions_new[np.newaxis, :], inputs_new[np.newaxis, :], state_new[np.newaxis, :], reward, done))

    def select_actions(self, avail_actions, inputs):
        agent_input = torch.from_numpy(inputs).to(self.device, dtype=torch.float)
        q_values, self.hidden_states = self.agent_net(agent_input, self.hidden_states)

        # select max actions among available actions
        mask = torch.from_numpy(avail_actions).to(self.device)
        q_values[mask == 0] = float('-inf')
        max_actions = torch.max(q_values, dim=1)[1]

        epsilon_choice = (torch.rand(max_actions.shape, device=self.device) < self.epsilon).long()
        random_actions = Categorical(mask).sample()
        picked_actions = epsilon_choice * random_actions + (1 - epsilon_choice) * max_actions

        return picked_actions.detach().cpu().numpy()

    def calc_total_q_values(self, batch_data):
        """step1: split the batch data and change the numpy data to tensor data """
        inputs, state, actions, inputs_new, avail_actions_new, state_new, reward, done = self._numpy_to_tensor_batch(batch_data)
        max_episode_len = state.shape[1]

        """step2: calculate the total q_values """
        # calc the q_cur and q_tar
        q_cur = []
        q_tar = []
        hidden_cur = self.agent_net.init_hidden().unsqueeze(0).expand(self.batch_size, self.n_agents, -1)
        hidden_tar = self.agent_net_tar.init_hidden().unsqueeze(0).expand(self.batch_size, self.n_agents, -1)
        for step in range(max_episode_len):
            input_cur = torch.index_select(inputs, 1, torch.tensor([step], device=self.device)).reshape(-1, self.input_shape)
            input_next = torch.index_select(inputs_new, 1, torch.tensor([step], device=self.device)).reshape(-1, self.input_shape)
            q_values_cur, hidden_cur = self.agent_net(input_cur, hidden_cur)
            q_values_tar, hidden_tar = self.agent_net_tar(input_next, hidden_tar)
            q_cur.append(q_values_cur.view(self.batch_size, self.n_agents, -1))
            q_tar.append(q_values_tar.view(self.batch_size, self.n_agents, -1))

        # concat over time and pick Q-values taken by each agent
        q_cur = torch.stack(q_cur, dim=1)
        q_cur = torch.gather(q_cur, -1, torch.transpose(actions, -1, -2))
        q_cur = torch.squeeze(q_cur).view(-1, 1, self.n_agents)

        # concat over time and mask unavailable actions
        q_tar = torch.stack(q_tar, dim=1)
        q_tar[~avail_actions_new] = float('-inf')
        q_tar = torch.max(q_tar, dim=-1)[0].detach().view(-1, 1, self.n_agents)

        """step3 cal the qtot_cur and qtot_tar by hyper_network"""
        qtot_cur = self.mixing_net(q_cur, self.hyper_net(state.view(self.batch_size*max_episode_len, -1)))
        qtot_tar = self.mixing_net(q_tar, self.hyper_net_tar(state_new.view(self.batch_size*max_episode_len, -1)))
        qtot_tar = reward + self.gamma * (1 - done) * qtot_tar

        return qtot_cur, qtot_tar
        
    def learn(self, epi_cnt, args):
        if self.epsilon > 0.05:
            self.epsilon -= args.anneal_par

        """ step1: get the batch data from the memory and change to tensor"""
        batch_data, num_diff_lens = self.memory.sample(self.batch_size)
        q, q_ = self.calc_total_q_values(batch_data)

        """ step2: cal the loss by bellman equation """
        q = q.view(self.batch_size, -1)
        q_ = q_.view(self.batch_size, -1)

        # delete the loss created by 0_padding data
        q_cur = q[0][:-num_diff_lens[0]]
        q_tar = q_[0][:-num_diff_lens[0]]
        for batch_cnt in range(1, self.batch_size):
            q_cur = torch.cat((q_cur, q[batch_cnt][:-num_diff_lens[batch_cnt]]), dim=0)
            q_tar = torch.cat((q_tar, q_[batch_cnt][:-num_diff_lens[batch_cnt]]), dim=0)

        loss = F.mse_loss(q_tar.detach(), q_cur)

        """ step3: loss backward """
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

        """ step4: update the tar and cur network """
        if epi_cnt > self.last_update_cnt \
                and (epi_cnt - self.last_update_cnt) % self.target_update_interval == 0:
            self.last_update_cnt = epi_cnt
            self.hyper_net_tar.load_state_dict(self.hyper_net.state_dict())
            self.agent_net_tar.load_state_dict(self.agent_net.state_dict())

        return loss

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.agent_net_tar, os.path.join(save_path, 'agent_net.pkl'))
        torch.save(self.hyper_net_tar, os.path.join(save_path, 'hyper_net.pkl'))

    def _numpy_to_tensor_batch(self, batch_data):
        np_inputs, np_state, np_actions, np_avail_actions_new, np_inputs_new, np_state_new, np_reward, np_done = batch_data

        inputs = torch.from_numpy(np_inputs).to(self.device, dtype=torch.float)
        state = torch.from_numpy(np_state).to(self.device, dtype=torch.float)
        actions = torch.from_numpy(np_actions).to(self.device, dtype=torch.long)
        inputs_new = torch.from_numpy(np_inputs_new).to(self.device, dtype=torch.float)
        avail_actions_new = torch.from_numpy(np_avail_actions_new).to(self.device, dtype=torch.uint8)
        state_new = torch.from_numpy(np_state_new).to(self.device, dtype=torch.float)
        reward = torch.from_numpy(np_reward).to(self.device, dtype=torch.float)
        done = torch.from_numpy(np_done).to(self.device, dtype=torch.float)

        return inputs, state, actions, inputs_new, avail_actions_new, state_new, reward, done
