import sys
sys.path.append('.')
import datetime
import argparse
import importlib
import os
from agent import Agent
from trainer import Trainer
from config import agent_config, trainer_config
from util.buffer import ReplayBuffer
from util.setting import set_device, set_global_seed
from util.logger import Logger
from torch.utils.tensorboard import SummaryWriter
from dynamic.transition_model import TransitionModel
import dsrl
import safety_gymnasium as gym

import numpy as np

TASK_DOMAIN = ['offlinepointgoal2', 'halfcheetah', 'hopper', 'walker2d']


class DummyStaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        return np.array([False]).repeat(len(obs))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="COMBO")
    parser.add_argument("--task", type=str, default="OfflinePointGoal2-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--logdir", type = str, default = 'log')

    return parser.parse_args()

def train(args = get_args()):
    set_device()
    set_global_seed(args.seed)
    # env
    print(f"task: {args.task}")
    env = gym.make(args.task, seed = args.seed)
    # env.seed(seed = args.seed)
    dataset = env.get_dataset()

    obs_space = env.observation_space
    act_space = env.action_space
    offline_buffer = ReplayBuffer(obs_space, act_space, buffer_size = len(dataset['observations']))
    # buffer
    offline_buffer.load_dataset(dataset)
    model_buffer = ReplayBuffer(obs_space, act_space, buffer_size = len(dataset['observations']))
    # agent
    agent = Agent(obs_space, act_space, **agent_config)

    # dynamic model
    task = None
    for key in args.task.split('-'):
        if key.lower() in TASK_DOMAIN:
            task = key.lower()
            break
    assert task != None

    print(task)

    # import_path = f"dynamic.static_fns.{task}"
    # static_fns = importlib.import_module(import_path).StaticFns
    # dummy static fns
    dummy_static_fns = DummyStaticFns()
    model_lr = trainer_config['model']['learning_rate']
    reward_penalty_coef = trainer_config['model']['reward_penalty_coef']
    cost_penalty_coef = trainer_config['model']['cost_penalty_coef']
    model = TransitionModel(obs_space, act_space, dummy_static_fns, model_lr, reward_penalty_coef, cost_penalty_coef, **trainer_config)
    # log
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    log_file = f'seed_{args.seed}_{t0}-{args.task.replace("-", "_")}_{args.algo}'
    log_path = os.path.join(args.logdir, args.task, args.algo, log_file)
    writer = SummaryWriter(log_path)
    logger = Logger(writer)
    # trainer
    trainer = Trainer(agent,
                      train_env = env, 
                      eval_env = env,
                      log = logger,
                      offline_buffer = offline_buffer,
                      model_buffer = model_buffer,
                      dynamic_model = model,
                      task = args.task, 
                      **trainer_config)
    # train
    trainer.train_dynamic()
    trainer.train()
    # trainer.save_video_demo(ite = args.algo + '_' + args.task)

if __name__ == '__main__':
    train()