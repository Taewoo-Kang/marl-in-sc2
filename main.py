from smac.env.pettingzoo import StarCraft2PZEnv
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser();
    parser.add_argument('--map', required=True, default='3m', help='select map in map_list')
    args = parser.parse_args()
    env = StarCraft2PZEnv.env(map_name=args.map)

    max_steps = 10000
    max_epi = 100

    steps = 0
    num_epi = 0
    total_rewards = 0

    while steps < max_steps and num_epi < max_epi:
        env.reset()

        for agent in env.agent_iter():
            env.render()
            obs, reward, done, _ = env.last()

            if done:
                action = None
            else:
                action = np.random.choice(np.flatnonzero(obs["action_mask"]))
            total_rewards += reward

            env.step(action)
            steps += 1

        num_epi += 1

    env.close()


if __name__ == "__main__":
    main()
