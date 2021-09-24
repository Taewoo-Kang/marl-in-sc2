import os
import numpy as np

from utils.logger import Logger
from algorithms.qmix import QmixNetwork


class Runner:
    def __init__(self, env, args):
        env.reset()
        self.renderer = env  # PettingZoo env for rendering
        self.env = env.env.env.env.env  # StarCraft2Env in smac

        env_info = self.env.get_env_info()
        state_shape = env_info['state_shape']
        obs_shape = env_info['obs_shape']
        n_actions = env_info['n_actions']
        self.n_agents = env_info['n_agents']

        self.logger = Logger(args)
        self.model_save_path = os.path.join(args.save_dir, "models", args.algorithm, args.start_time)

        if args.algorithm == 'QMIX':
            self.controller = QmixNetwork(state_shape, obs_shape, n_actions, self.n_agents, args)
        elif args.algorithm == 'DOP':
            pass

    def train(self, args):
        for epi_cnt in range(args.max_episode):
            episode_reward = self._run_episode(test_mode=False)
            if epi_cnt > args.learning_start_episode:
                loss = self.controller.learn(epi_cnt, args)
                if epi_cnt % args.log_interval == 0:
                    self.logger.log("loss", loss, epi_cnt)

            if epi_cnt > 0 and epi_cnt % args.log_interval == 0:
                stats = self.env.get_stats()
                # self._log_stats(stats, epi_cnt)
                self.logger.log("win_rate", stats["win_rate"], epi_cnt)
                self.logger.log("episode_reward", episode_reward, epi_cnt)

            if epi_cnt > 0 and epi_cnt % args.eval_interval == 0:
                self.evaluate(args, epi_cnt)

            if epi_cnt > args.start_save_model and epi_cnt % args.save_model_interval:
                save_path = os.path.join(self.model_save_path, str(epi_cnt))
                self.controller.save_model(save_path)

        self.env.close()
        self.logger.console_logger.info("Training is done!")

    def evaluate(self, args, epi_cnt):
        rewards = []
        battles_won = []
        for _ in range(args.eval_episode_num):
            episode_reward = self._run_episode(test_mode=True)
            rewards.append(episode_reward)
            battles_won.append(self.env.win_counted)
        self.logger.log("eval_reward_mean", np.mean(rewards), epi_cnt)
        self.logger.log("eval_battles_won_mean", np.mean(battles_won), epi_cnt)
        self.env.close()

    def test(self, args):
        pass

    def _run_episode(self, test_mode=False):
        self.env.reset()
        done = False
        episode_reward = 0
        self.controller.init_hidden(batch_size=1)  # episode batch is 1
        if not test_mode:
            self.controller.memory.create_new_episode()

        while not done:
            if test_mode:
                self.renderer.render()

            state = self.env.get_state()
            obs = self.env.get_obs()
            avail_actions = np.array(self.env.get_avail_actions())
            agent_input = self._make_agent_input(obs, self.env.last_action)

            actions = self.controller.select_actions(avail_actions, agent_input)
            reward, done, _ = self.env.step(actions)
            episode_reward += reward

            if not test_mode:
                state_new = self.env.get_state()
                obs_new = self.env.get_obs()
                avail_actions_new = np.array(self.env.get_avail_actions())
                agent_input_new = self._make_agent_input(obs_new, self.env.last_action)

                self.controller.save_memory(agent_input, state, actions.reshape(1, -1),
                                            avail_actions_new, agent_input_new, state_new, reward, done)

        # TODO: Why PyMARL select actions in the last state when after the environment is done?

        return episode_reward

    def _make_agent_input(self, obs, last_action):
        """
        Agent input consists of current observation 'o_t', last selected action 'u_(t-1)' and agent ID.
        Since stochastic policies during training, actual selected action is provided as agent input.
        If weights are shared across the agent networks, agent IDs are included to allow for heterogeneous policies.
        """
        return np.concatenate([np.eye(self.n_agents), obs, last_action], axis=1)

    def _log_stats(self, stats, epi_cnt):
        for k, v in stats.items():
            self.logger.log(k, v, epi_cnt)
        self.logger.log("battles_won_mean", stats["battles_won"] / stats["battles_game"], epi_cnt)
