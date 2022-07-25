# stable baseline: OpenAI

from collections import defaultdict
import numpy as np
import os
import json

from datetime import datetime as dt
import argparse
import torch

from RL.environment import BUNDLERecommendEnv
from RL.optimization import HierPPO
from RL.policy import ManagerPolicy, SidePolicy

from utils import random_seed, last_commit_msg, save_dependencies

import warnings
warnings.filterwarnings("ignore")

def main():
    p = argparse.ArgumentParser()
    # dataset
    p.add_argument('--data', type=str, default='steam', 
                   help='data') 
    # model
    p.add_argument('--model', type=str, default='Bunt', 
                   help='Bunt or MaskedBunt or BGN')
    p.add_argument('--embed_size', type=int, default=32,
                   help='embed size for items')
    p.add_argument('--hidden_size', type=int, default=32,
                   help='hidden size for BGN')
    p.add_argument('--n_layers', type=int, default=1, 
                   help='layer number of model') 
    p.add_argument('--n_heads', type=int, default=1, 
                   help='head number of model') 
    p.add_argument('--policy_strategy', type=str, default='learn',
                   help='Selected from "random" or "learn"')
    # environment
    p.add_argument('--seed', type=int, default=1, 
                  help='random seed.')
    p.add_argument('--epochs', type=int, default=50000, 
                  help='the number of RL train epoch')
    p.add_argument('--batch_size', type=int, default=128, 
                  help='batch size.')
    p.add_argument('--max_run', type=int, default=10, 
                  help='max conversation runs')
    p.add_argument('--device', type=str, default='cuda:0', 
                  help='cuda:x or cpu')
    p.add_argument('--reward', type=str, default='prec', 
                  help='reward types')
    p.add_argument('--dropout', type=float, default=0.2, 
                  help='Dropout prob')
    p.add_argument('--ask_k', type=int, default=1, 
                  help='k cate(s) or attr(s) to ask')
    p.add_argument('--delta_len', type=int, default=2, 
                  help='d items to recommend')
    # online optimization
    p.add_argument('--n_steps', type=int, default=512, 
                  help='rollout collection steps') 
    p.add_argument('--n_eval_samples', type=int, default=512, 
                  help='users sampled for validation evaluation') 
    p.add_argument('--gamma', type=float, default=0.99, 
                  help='reward discount factor.')
    p.add_argument('--gae_lambda', type=float, default=0.95, 
                  help='gae factor') 
    p.add_argument('--clip_range', type=float, default=0.2, 
                  help='clip range for PPO') 
    p.add_argument('--vf_coef', type=float, default=0.2, 
                  help='value function coefficient')
    p.add_argument('--n_epochs', type=int, default=10, 
                  help='the number of RL train epoch')
    p.add_argument('--max_grad_norm', type=float, default=0.5, 
                  help='max gradient norm for RL train epoch')
    # evaluation
    p.add_argument('--eval_max_run', type=int, default=10, 
                  help='max eval conversation runs')
    p.add_argument('--command', type=int, default=1, 
                  help='select state vector')
    p.add_argument('--trivial_test', type=int, default=0,
                  help='whether test all-rec and random setting')
    # logging
    p.add_argument('--verbose', type=int, default=1000, 
                  help='the number of epochs to save RL model and metric')  
    p.add_argument('--pretrained_weights', type=str, default=None, 
                  help='path for pre-trained bundle recommender')
    p.add_argument('--ckpt_dir', type=str, default='test', 
                  help='checkpoint saving directory')
    p.add_argument('--load_ckpt_dir', type=str, default=None, 
                  help='checkpoint loading directory')
    args = p.parse_args()
    # ablation

    random_seed(args.seed)

    # logging folder
    branch, commit = last_commit_msg()
    ckpt_dir = os.path.join('checkpoints', branch, commit, args.ckpt_dir, f'seed_{args.seed}_{dt.now().strftime("%Y-%m-%d-%H-%M-%S")}')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    with open(os.path.join(ckpt_dir, "args.log"), "w") as f:
        f.write(json.dumps(vars(args), indent=2)) 
    save_dependencies(ckpt_dir)
    print(f"set ckpt as {ckpt_dir}")

    # initilization
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    args.ckpt_dir = ckpt_dir
    args.side_policy_class = SidePolicy
    args.conv_policy_class = ManagerPolicy

    # env
    env = BUNDLERecommendEnv(args)

    # ckpt test
    agent = None
    if args.load_ckpt_dir is not None:  
        env_test = BUNDLERecommendEnv(args, mode='test')
        env_test.evaluate(agent, 2, save=os.path.join(args.load_ckpt_dir, "all-rec")) 
        env_test.evaluate(HierPPO(env=env, args=args), 3, save=os.path.join(args.load_ckpt_dir, "learn"))
        exit()

    # trivial test
    if args.trivial_test:
        env_test = BUNDLERecommendEnv(args, mode='test')
        res = env_test.evaluate(agent, 1, save=os.path.join(args.ckpt_dir, "random"))
        with open(os.path.join(args.ckpt_dir, "random.log"), "w") as f:
            json.dump(res, f)
        res = env_test.evaluate(agent, 2, save=os.path.join(args.ckpt_dir, "all-rec"))
        with open(os.path.join(args.ckpt_dir, "all-rec.log"), "w") as f:
            json.dump(res, f)
        exit()

    # training
    agent = HierPPO(env=env, args=args).learn(args.epochs, n_eval_episodes=1, eval_log_path=args.ckpt_dir)

    # final test
    args.load_ckpt_dir = args.ckpt_dir
    env_test = BUNDLERecommendEnv(args, mode='test')
    res = env_test.evaluate(agent, 3, save=os.path.join(args.ckpt_dir, "learn"))
    with open(os.path.join(args.ckpt_dir, "test.log"), "w") as f:
        json.dump(res, f)
    print(f"save results into ... {ckpt_dir}")

if __name__ == '__main__':
    main()