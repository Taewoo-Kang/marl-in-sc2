import torch
import argparse


def parse_args():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser("Multi-Agent Reinforcement Learning Algorithms in StarCraft II Environment")

    # environment
    parser.add_argument("--map_name", type=str, default="3m", help="name of the scenario script")
    parser.add_argument("--difficulty", type=str, default='3', help="difficulty of the scenario script")

    # runner
    parser.add_argument("--test_mode", type=bool, default=False, help="train or test")
    parser.add_argument("--algorithm", type=str, default="QMIX", help="multi-agent algorithm name")
    parser.add_argument("--max_episode", type=int, default=40000, help="maximum episode length")
    parser.add_argument("--per_episode_max_len", type=int, default=120, help="maximum episode length")
    parser.add_argument("--learning_start_episode", type=int, default=1000, help="learning start episode")
    parser.add_argument("--learning_freq", type=int, default=1, help="learning frequency")
    parser.add_argument("--log_interval", type=int, default=100, help="the num for print log")
    parser.add_argument("--eval_interval", type=int, default=500, help="evaluation frequency")
    parser.add_argument("--eval_episode_num", type=int, default=10, help="the num for evaluation")

    # model hpyerparameters
    parser.add_argument("--agent_hidden_dim", type=list, default=64)
    parser.add_argument("--hyper_embed_dim", type=int, default=64)
    parser.add_argument("--mixing_embed_dim", type=int, default=32)

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--batch_size", type=int, default=32, help="number of episodes to optimize at the same time")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--anneal_par", type=float, default=0.0004075, help="annealing e-greedy")
    parser.add_argument("--epsilon", type=float, default=1.0, help="the init par for e-greedy")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate for adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=10, help="max gradient norm for clip")
    parser.add_argument("--target_update_interval", type=int, default=200, help="episode for update target net")
    parser.add_argument("--memory_size", type=int, default=5000, help="number of data stored in the memory")

    # checkpointing
    parser.add_argument("--save_dir", type=str, default='')
    parser.add_argument("--start_save_model", type=int, default=10000, help="saving the model")
    parser.add_argument("--save_model_interval", type=int, default=1000)

    return parser.parse_args()
