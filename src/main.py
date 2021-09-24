import os
from os.path import dirname, abspath
from datetime import datetime

from smac.env.pettingzoo import StarCraft2PZEnv
from utils.argparser import parse_args
from run import Runner


def main():
    args = parse_args()
    if args.save_dir == '':
        args.save_dir = os.path.join(dirname(dirname(abspath(__file__))), "results")
    args.start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    env = StarCraft2PZEnv.env(map_name=args.map_name, difficulty=args.difficulty)
    runner = Runner(env, args)
    if args.test_mode:
        runner.test(args)
    else:
        runner.train(args)


if __name__ == "__main__":
    main()
