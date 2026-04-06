import gymnasium as gym
import torch
import torch.nn as nn

import math
import copy
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import distributions as pyd
import torch.optim as optim
from torch.distributions import Categorical

import DiffusionPolicy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-8
def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk

class PolicyBase(nn.Module):
    def __init__(self):
        pass

    def forward(self, state):
        """ 
        Input: state to perform forward inference
        Output: (action_sample from policy distribution, log_prob of sampled action)
        """
        pass
    
    def log_prob(self, state, action):
        """
        Input: state to perform forward inference, action to evaluate log probability
        Output: Log probability of action distribution under the policy distribution using state
        """
        pass
    
class PolicyGaussian(nn.Module):       
    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2):
        super(PolicyGaussian, self).__init__()
        self.trunk = mlp(num_inputs, hidden_dim, num_outputs*2, hidden_depth)

    def forward(self, state):
        outs = self.trunk(state)
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1)
        std = torch.exp(logstd) + EPS
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1)
        ac = ac_dist.sample()
        return ac, ac_dist.log_prob(ac)

    def log_prob(self, state, action):
        outs = self.trunk(state)
        mu, logstd = torch.split(outs, outs.shape[-1] // 2, dim=-1)
        std = torch.exp(logstd) + EPS
        ac_dist = torch.distributions.Independent(torch.distributions.Normal(mu, std), reinterpreted_batch_ndims=1)
        return ac_dist.log_prob(action)

class PolicyAutoRegressiveModel(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_dim=65, hidden_depth=2, num_buckets=10, ac_low=-1, ac_high=1):
        super(PolicyAutoRegressiveModel, self).__init__()
        self.eps = 1e-8
        self.trunks = nn.ModuleList([mlp(num_inputs, hidden_dim, num_buckets, hidden_depth)] \
                        + [mlp(num_inputs + j + 1, hidden_dim, num_buckets, hidden_depth) for j in range(num_outputs - 1)])
        self.num_dims = num_outputs
        self.ac_low = torch.tensor(ac_low).to(device)
        self.ac_high = torch.tensor(ac_high).to(device)
        self.num_buckets = num_buckets
        self.bucket_size = torch.tensor((ac_high - ac_low) / num_buckets).to(device)
         
    def discretize(self, ac):
        bucket_idx = (ac - self.ac_low) // (self.bucket_size + self.eps)
        return torch.clip(bucket_idx, 0, self.num_buckets - 1)
    
    def undiscretize(self, bucket_idx, dimension):
        return_val = bucket_idx[:, None]*self.bucket_size + self.ac_low + self.bucket_size*0.5
        return return_val[:, dimension]

    def forward(self, state):
        vals = []
        log_probs = 0
        for j in range(self.num_dims):
            #========== TODO: start ==========
            # Here, we want to predict each action one dimension at a time.
            # For each action dimension j, concatenate state with all the previous action dimensions (0...j-1), pass it through
            # the respective MLP (i.e. self.trunks[j]) to get a logit. Use the logit to create a categorical distribution (torch.Categorical).
            # Sample from this distribution and get that sample's log probability. Add the log probability
            # to the running log_probs and undiscretize the sample add append it to vals.
            # Important - use previous *sampled* actions
            pass
            #========== TODO: end ==========
        vals = torch.cat(vals, dim=-1)
        return vals, log_probs
    
    def log_prob(self, state, action):
        log_prob = 0.
        ac_discretized = self.discretize(action)
        for j in range(self.num_dims):
            #========== TODO: start ==========
            # Here, want to get log prob of action given state under the current autoregressive model.
            # For each action dimension j, concatenate state with all the previous action dimensions (0...j-1), pass it through
            # the respective MLP (i.e. self.trunks[j]) to get a logit. Use the logit to create a categorical distribution.
            # Get the log prob of the respective discretized action (i.e. ac_discretized[:, j]) and add it to the running log_prob.
            # Important - use previous actions from the action variable, *not* sampled actions
            pass
            #========== TODO: end ==========

        return log_prob
    
def rollout(
        env,
        agent,
        agent_name, # Should be bc, dagger, pg
        episode_length=math.inf,
        render=False,
):
    if agent_name == "diffusion":
        return rollout_diffusion(env, agent, episode_length, render)
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []

    entropy = None
    log_prob = None
    agent_info = None
    path_length = 0

    o, _ = env.reset()
    if render:
        env.render()

    while path_length < episode_length:
        o_for_agent = o.copy()
        o_for_agent = torch.from_numpy(o_for_agent[None]).to(device).float()
        action, _ = agent(o_for_agent) # TODO: May need to convert to numpy
        action = action.cpu().detach().numpy().squeeze()
        # Step the simulation forward
        next_o, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Render the environment
        if render:
            env.render()

        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        path_length += 1
        if done:
            break
        o = next_o

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)
    
    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images = np.array(images)
    )
    
    
def rollout_diffusion(
        env,
        agent,
        episode_length=math.inf,
        render=False,):
     # # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []
    obs, _ = env.reset()
    agent.add_obs(obs.copy())
    # save visualization and rewards
    rewards = list()
    done = False
    step_idx = 0

    while not done:
        action, _ = agent.get_action()
        action = action.cpu().detach().numpy()
        # execute action_horizon number of steps
        # without replanning
        for i in range(len(action)):
            # stepping env
            obs, reward, terminated, truncated, _ = env.step(action[i])
            done = terminated or truncated

            if render:
                env.render()
            agent.add_obs(obs.copy())
            # and reward/vis
            rewards.append(reward)

            # update progress bar
            step_idx += 1
            if step_idx > episode_length:
                done = True
                break
            if reward > 0:
                done = True
                break
        
    return dict(
        observations=[],
        next_observations=[],
        actions=actions,
        rewards=rewards,
        dones=[reward>0],
        # dones=np.array(dones).reshape(-1, 1),
        images = np.array(images)
    )

def generate_paths(env, expert_policy, episode_length, num_paths, file_path):
    # Initial data collection
    paths = []
    for j in range(num_paths):
        path = rollout(
            env,
            expert_policy,
            agent_name='bc',
            episode_length=episode_length,
            render=False)
        print("return is " + str(path['rewards'].sum()))
        paths.append(path)

    with open(file_path, 'wb') as fp:
        pickle.dump(paths, fp)
    print('Paths has been save to the file')

def get_expert_data(file_path):
    with open(file_path, 'rb') as fp:
        expert_data = pickle.load(fp)
    print('Imported Expert data successfully')
    return expert_data

def relabel_action(path, expert_policy):
    for i in range(len(path)):
        observation = path['observations']
        expert_action = expert_policy.get_action(observation)
        path['actions'] = expert_action[0]
    return path

def combine_sample_trajs(sample_trajs):
    assert len(sample_trajs) > 0

    my_dict = {k: [] for k in sample_trajs[0]}
    sample_trajs[0].keys()
    for sample_traj in sample_trajs:
        for key, value in sample_traj.items():
            my_dict[key].append(value)
    
    for key, value in my_dict.items():
        my_dict[key] = np.array(value)

    return my_dict
