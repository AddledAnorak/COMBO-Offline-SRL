import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.agents import BaseAgent
from util.buffer import ReplayBuffer
from util.net import MLP, get_optimizer
from util.policy_net import GaussPolicy, CategoricalPolicy
from util.value_net import QNet
from util.setting import DEVICE
from util.helper import soft_update_network
from copy import deepcopy
from operator import itemgetter
import safety_gymnasium as gym
import numpy as np

from util.lagrange import Lagrange

class Agent(BaseAgent):
    def __init__(self, obs_space, action_space, **kwargs) -> None:
        BaseAgent.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space

        self.obs_dim = obs_space.shape[0]

        hidden_dims_pi = kwargs['policy_net']['hidden_dims']
        act_fun_pi = kwargs['policy_net']['act_fun']
        out_act_fun_pi = kwargs['policy_net']['out_act_fun']
        hidden_dims_q = kwargs['q_net']['hidden_dims']
        act_fun_q = kwargs['q_net']['act_fun']
        out_act_fun_q = kwargs['q_net']['out_act_fun']

        self.discrete = False
        # initialize network
        # NOTE: does not work for discrete action space!
        self.out_dim = action_space.shape[0]
        out_std = kwargs['policy_net']['out_std']
        conditioned_std = kwargs['policy_net']['conditioned_std']
        reparameter = kwargs['policy_net']['reparameter']
        log_std = kwargs['policy_net']['log_std']
        log_std_min = kwargs['policy_net']['log_std_min']
        log_std_max = kwargs['policy_net']['log_std_max']
        stable_log_prob = kwargs['policy_net']['stable_log_prob']

        self.policy_net = GaussPolicy(self.obs_dim, action_space, hidden_dims_pi, act_fun_pi, out_act_fun_pi, 
                                        out_std, conditioned_std, reparameter, log_std, log_std_min, log_std_max, stable_log_prob).to(DEVICE)
        self.q1 = QNet(self.obs_dim + self.out_dim, 1, hidden_dims_q, act_fun_q, out_act_fun_q).to(DEVICE)
        self.q2 = QNet(self.obs_dim + self.out_dim, 1, hidden_dims_q, act_fun_q, out_act_fun_q).to(DEVICE)
        self.target_q1 = deepcopy(self.q1)
        self.target_q2 = deepcopy(self.q2)

        self.cost_limit = kwargs['lagrange']['cost_limit']
        self.lambda_optimizer = kwargs['lagrange']['lambda_optimizer']
        self.lambda_lr = kwargs['lagrange']['lambda_lr']
        self.lambda_init = kwargs['lagrange']['lambda_init']
        self.lambda_upper_bound = kwargs['lagrange']['lambda_upper_bound']

        self.lagrange = Lagrange(self.cost_limit, self.lambda_init, self.lambda_lr, self.lambda_optimizer, self.lambda_upper_bound)

        # create cost q networks
        self.cost_q = QNet(self.obs_dim + self.out_dim, 1, hidden_dims_q, act_fun_q, out_act_fun_q).to(DEVICE)
        self.target_cost_q = deepcopy(self.cost_q)
        
        self.networks = {
            'policy_net' : self.policy_net,
            'critic_net1' : self.q1,
            'critic_net2' : self.q2,
            'cost_critic_net' : self.cost_q
        }

        # optimizer
        self.q1_opt = get_optimizer(kwargs['q_net']['opt_name'], self.q1, kwargs['q_net']['learning_rate'])
        self.q2_opt = get_optimizer(kwargs['q_net']['opt_name'], self.q2, kwargs['q_net']['learning_rate'])
        self.cost_q_opt = get_optimizer(kwargs['q_net']['opt_name'], self.cost_q, kwargs['q_net']['learning_rate'])
        self.policy_opt = get_optimizer(kwargs['policy_net']['opt_name'], self.policy_net, kwargs['policy_net']['learning_rate'])

        # entropy
        self.alpha = kwargs['alpha']
        self.auto_alpha = kwargs['entropy']['auto_alpha']
        if self.auto_alpha:
            self.target_entropy = - np.prod(action_space.shape).item()
            self.log_alpha = nn.Parameter(torch.zeros(1, device = DEVICE), requires_grad = True)
            self.alpha = self.log_alpha.detach().exp()
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr = kwargs['entropy']['learning_rate'])
        self.gamma = kwargs['gamma']
        self.tau = kwargs['tau']

        # cql
        self.rand_times = kwargs['rand_times']
        self.cql_temp = kwargs['cql_temp']
        self.cql_weight = kwargs['cql_weight']
        self.importance_sample = kwargs['importance_sample']

    def update_target_network(self):
        soft_update_network(self.target_q1, self.q1, self.tau)
        soft_update_network(self.target_q2, self.q2, self.tau)
        soft_update_network(self.target_cost_q, self.cost_q, self.tau)
        
    def update(self, batch_data):
        obs = batch_data['obs']
        actions = batch_data['action']
        next_obs = batch_data['next_obs']
        rewards = batch_data['reward']
        costs = batch_data['cost']
        dones = batch_data['done']

        # critic loss
        cur_state_q1_values = self.q1(torch.cat([obs, actions], dim = 1))
        cur_state_q2_values = self.q2(torch.cat([obs, actions], dim = 1))

        # cost critic loss
        cur_state_cost_q_values = self.cost_q(torch.cat([obs, actions], dim = 1))
        
        with torch.no_grad():
            next_state_action, next_state_log_pi = \
                itemgetter('action', 'log_prob')(self.policy_net.sample(next_obs))
            
            next_state_q1_value = self.target_q1(torch.cat([next_obs, next_state_action], dim = 1))
            next_state_q2_value = self.target_q2(torch.cat([next_obs, next_state_action], dim = 1))
            next_state_min_q = torch.min(next_state_q1_value, next_state_q2_value)
            target_q = (next_state_min_q - self.alpha * next_state_log_pi)
            target_q  = rewards + self.gamma * (1.0 - dones) * target_q

            next_state_cost_q_value = self.target_cost_q(torch.cat([next_obs, next_state_action], dim = 1))
            target_cost_q = costs + self.gamma * (1.0 - dones) * next_state_cost_q_value # no regularization here
        
        q1_td_loss = F.mse_loss(cur_state_q1_values, target_q)
        q2_td_loss = F.mse_loss(cur_state_q2_values, target_q)
        cost_q_td_loss = F.mse_loss(cur_state_cost_q_values, target_cost_q)
        # q1_loss = q1_td_loss
        # q2_loss = q2_td_loss
        # self.q1_opt.zero_grad()
        # q1_td_loss.backward()
        # self.q1_opt.step()
        # self.q2_opt.zero_grad()
        # q2_td_loss.backward()
        # self.q2_opt.step()

        # normal loss for cost critic
        self.cost_q_opt.zero_grad()
        cost_q_td_loss.backward()
        self.cost_q_opt.step()

        # cql loss
        batch_size = batch_data['obs'].shape[0]
        scale = self.policy_net.action_scale[0]
        random_actions = torch.rand(batch_size * self.rand_times, batch_data['action'].shape[-1]).uniform_(-scale, scale).to(DEVICE)
        obs_repeat = obs.unsqueeze(0).repeat(1, self.rand_times, 1).view(batch_size * self.rand_times, obs.shape[-1])
        next_obs_repeat = next_obs.unsqueeze(0).repeat(1, self.rand_times, 1).view(batch_size * self.rand_times, next_obs.shape[-1])
        random_q1 = self.q1(torch.cat([obs_repeat, random_actions], dim = 1))
        random_q2 = self.q2(torch.cat([obs_repeat, random_actions], dim = 1))
        cur_actions, cur_log = itemgetter('action', 'log_prob')(self.policy_net.sample(obs_repeat))
        new_cur_actions, next_log = itemgetter('action', 'log_prob')(self.policy_net.sample(next_obs_repeat))
        cur_state_q1 = self.q1(torch.cat([obs_repeat, cur_actions], dim = -1))
        cur_state_q2 = self.q2(torch.cat([obs_repeat, cur_actions], dim = -1))
        cur_next_state_q1 = self.q1(torch.cat([obs_repeat, new_cur_actions], dim = -1))
        cur_next_state_q2 = self.q2(torch.cat([obs_repeat, new_cur_actions], dim = -1))
        sample_q1 = torch.cat([
            random_q1,
            cur_state_q1_values.unsqueeze(0).repeat(1, self.rand_times, 1).view(batch_size * self.rand_times, -1),
            cur_next_state_q1,
            cur_state_q1 
        ], dim = -1)
        sample_q2 = torch.cat([
            random_q2,
            cur_state_q2_values.unsqueeze(0).repeat(1, self.rand_times, 1).view(batch_size * self.rand_times, -1),
            cur_next_state_q2,
            cur_state_q2 
        ], dim = -1)
        if self.importance_sample:
            random_density = np.log(0.5 ** actions.shape[-1])
            sample_q1 = torch.cat([
                random_q1 - random_density,
                cur_next_state_q1 - next_log.detach(),
                cur_state_q1 - cur_log.detach()
            ], dim = -1)
            sample_q2 = torch.cat([
                random_q2 - random_density,
                cur_next_state_q2 - next_log.detach(),
                cur_state_q2 - cur_log.detach()
            ], dim = -1)
        min_q1_loss = torch.logsumexp(sample_q1 / self.cql_temp, dim = -1).mean() * self.cql_temp - cur_state_q1_values.mean()
        min_q2_loss = torch.logsumexp(sample_q2 / self.cql_temp, dim = -1).mean() * self.cql_temp - cur_state_q2_values.mean()
        q1_loss = q1_td_loss + min_q1_loss * self.cql_weight
        q2_loss = q2_td_loss + min_q2_loss * self.cql_weight

        self.q1_opt.zero_grad()
        q1_loss.backward(retain_graph = True)
        self.q1_opt.step()
        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # actor loss
        cur_state_action, cur_state_log_pi = \
            itemgetter('action', 'log_prob')(self.policy_net.sample(obs))
        cur_state_action_q1 = self.q1(torch.cat([obs, cur_state_action], dim = 1))
        cur_state_action_q2 = self.q2(torch.cat([obs, cur_state_action], dim = 1))
        cur_state_action_q = torch.min(cur_state_action_q1, cur_state_action_q2)
        actot_loss = ((self.alpha * cur_state_log_pi) - cur_state_action_q).mean()
        # cost actor loss
        cur_state_cost = self.cost_q(torch.cat([obs, cur_state_action], dim = 1))
        actor_cost_loss = (self.lagrange.lagrangian_multiplier.item() * cur_state_cost).mean()
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (cur_state_log_pi + self.target_entropy).detach()).mean()
            alpha_loss_value = alpha_loss.detach().cpu().item()
            self.alpha_opt.zero_grad()
        else:
            alpha_loss = 0.0
            alpha_loss_value = 0.0
        self.policy_opt.zero_grad()
        (actor_cost_loss + actot_loss + alpha_loss).backward()
        self.policy_opt.step()
        if self.auto_alpha:
            self.alpha_opt.step()
            self.alpha = self.log_alpha.detach().exp() 
        self.update_target_network()

        # udpate the lagrange multiplier
        self.lagrange.update_lagrange_multiplier(cur_state_cost.mean().item())

        # return {
        #     'loss/policy_reward' : actot_loss.item(),
        #     'loss/policy_cost' : actor_cost_loss.item(),
        #     'loss/q1' : q1_loss.item(),
        #     'loss/q2' : q2_loss.item(),
        #     'loss/cost_critic' : cost_q_td_loss.item(),
        #     'loss/alpha' : alpha_loss_value,
        #     'misc/entroy_alpha' : self.alpha.item(),
        # }

        return {
            'policy/net_loss': (actor_cost_loss + actot_loss + alpha_loss).item(),
            'policy/cost_loss': actor_cost_loss.item(),
            'policy/actor_loss': actot_loss.item(),
            'policy/alpha_loss': alpha_loss_value,
            'q1/net_loss': q1_loss.item(),
            'q1/td_loss': q1_td_loss.item(),
            'q1/min_loss': min_q1_loss.item(),
            'q2/net_loss': q2_loss.item(),
            'q2/td_loss': q2_td_loss.item(),
            'q2/min_loss': min_q2_loss.item(),
            'cost_critic/td_loss': cost_q_td_loss.item(),
            'params/alpha': self.alpha.item(),
            'params/lambda': self.lagrange.lagrangian_multiplier.item(),
            'params/cql_weight': self.cql_weight
        }
    

    def choose_action(self, state, deterministic=True):
        flag = False
        if len(state.shape) == 1:
            state = [state]
            flag = True
        if type(state) != torch.Tensor:
            state = torch.FloatTensor(np.array(state)).to(DEVICE)
        action, log_prob = itemgetter('action', 'log_prob')(self.policy_net.sample(state, deterministic))
        if flag:
            action = action[0]
            log_prob = log_prob[0]
        return {
            'action' : action.detach().cpu().numpy(),
            'log_prob' : log_prob
        }
    

    def save_agent(self, info):
        # save all network weights by creating subfolders
        save_dir = os.path.join('./log/', 'models_pt')
        save_dir = os.path.join(save_dir, f'ite_{info}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(save_dir, network_name + ".pt")
            torch.save(network.state_dict(), save_path)


    def load_agent(self, info):
        save_dir = os.path.join('./log/', 'models_pt')
        save_dir = os.path.join(save_dir, f'ite_{info}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for network_name, network in self.networks.items():
            save_path = os.path.join(save_dir, network_name + ".pt")
            network.load_state_dict(torch.load(save_path))