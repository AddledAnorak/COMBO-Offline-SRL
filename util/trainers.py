from abc import ABC, abstractmethod
import numpy as np
import os
import cv2
from util.setting import LOG_PATH
import gym
from agent import Agent
class BaseTrainer(ABC):
    def __init__(self, 
                 agent: Agent, 
                 train_env,
                 eval_env,
                 log,
                 max_train_epoch,
                 max_traj_len,
                 eval_interval,
                 log_interval,
                 eval_traj_num,
                 **kwargs
                ) -> None:

        self.agent = agent
        self.train_env = train_env
        self.eval_env = eval_env
        self.trained_epochs = 0
        self.max_train_epoch = max_train_epoch
        self.max_traj_len = max_traj_len
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.eval_traj_num = eval_traj_num
        self.log = log

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    def eval(self):
        traj_returns = []
        traj_costs = []
        traj_lengths = []
        for cnt in range(self.eval_traj_num):
            traj_return = 0
            traj_cost = 0
            traj_length = 0
            state, _ = self.eval_env.reset()
            for step in range(self.max_traj_len):
                action = self.agent.choose_action(state, deterministic = True)['action']
                # if len(action) == 1 and type(self.eval_env.action_space) == gym.spaces.discrete.Discrete :
                #     action = action[0]
                next_state, reward, cost, truncated, terminated, _ = self.eval_env.step(action)
                done = truncated or terminated
                state = next_state
                traj_length += 1
                traj_return += reward
                traj_cost += cost
                if done:
                    break
                else:
                    state = next_state
            traj_lengths.append(traj_length)
            traj_returns.append(traj_return)
            traj_costs.append(traj_cost)
        return {
            'performance/eval_return' : np.mean(traj_returns),
            'performance/eval_length' : np.mean(traj_lengths), 
            'performance/eval_cost' : np.mean(traj_costs),
        }
    
    # def save_video_demo(self, ite, width=256, height=256, fps=30):
    #     video_demo_dir = os.path.join(LOG_PATH,"demos")
    #     if not os.path.exists(video_demo_dir):
    #         os.makedirs(video_demo_dir)
    #     video_size = (height, width)
    #     video_save_path = os.path.join(video_demo_dir, "ite_{}.mp4".format(ite))

    #     #initilialize video writer
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #     video_writer = cv2.VideoWriter(video_save_path, fourcc, fps, video_size)

    #     #rollout to generate pictures and write video
    #     state = self.eval_env.reset()
    #     img = self.eval_env.render(mode="rgb_array")
    #     video_writer.write(img)
    #     for step in range(self.max_traj_len):
    #         action = self.agent.choose_action(state)['action']
    #         if len(action) == 1 and type(self.eval_env.action_space) == gym.spaces.discrete.Discrete :
    #             action = action[0]
    #         next_state, reward, done, _ = self.eval_env.step(action)
    #         state = next_state
    #         img = self.eval_env.render(mode="rgb_array", width=width, height=height)
    #         video_writer.write(img)
    #         if done:
    #             break
                
    #     video_writer.release()
                
