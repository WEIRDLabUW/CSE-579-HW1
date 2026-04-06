import DiffusionPolicy
import gymnasium as gym
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import argparse

from utils import generate_paths, get_expert_data, PolicyGaussian, PolicyAutoRegressiveModel

from bc import simulate_policy_bc
from dagger import simulate_policy_dagger
import pytorch_utils as ptu
from evaluate import evaluate
from reach_goal.envs.pointmaze_env import PointMazeEnv
from reach_goal.envs.pointmaze_expert import WaypointController

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device', device)

torch.manual_seed(0)
import random
random.seed(0)
np.random.seed(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='reacher', help='choose env (reacher/pointmaze)')
    parser.add_argument('--train', type=str, default='behavior_cloning', help='choose training method (behavior cloning/dagger)')
    parser.add_argument('--policy', type=str, default='gaussian', help='choose policy class (gaussian/autoregressive)')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--render',  action='store_true', default=False)
    args = parser.parse_args()
    # Get the expert data
    if args.env == 'reacher':
        file_path = 'data/reacher_expert_data.pkl'
    elif args.env == 'pointmaze':
        file_path = 'data/pointmaze_expert_data.pkl'
    else:
        raise ValueError('Invalid environment')
    expert_data = get_expert_data(file_path)

    flattened_expert = {'observations': [], 
                        'actions': []}
    
    for expert_path in expert_data:
        for k in flattened_expert.keys():
            flattened_expert[k].append(expert_path[k])

    for k in flattened_expert.keys():
        flattened_expert[k] = np.concatenate(flattened_expert[k])
    
    # Define environment
    if args.env == 'reacher':
        env = gym.make("Reacher-v4", render_mode='human' if args.render else None)
    elif args.env == 'pointmaze':
        env = PointMazeEnv(render_mode='human' if args.render else 'rgb_array')
    else:
        raise ValueError('Invalid environment')

    # Define policy
    hidden_dim = 128
    hidden_depth = 2
    obs_size = env.observation_space.shape[0]
    ac_size = env.action_space.shape[0]
    ac_margin = 0.1

    if args.policy == 'gaussian':
        policy = PolicyGaussian(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        policy.to(device)
    elif args.policy == 'autoregressive':
        num_buckets = 10
        policy = PolicyAutoRegressiveModel(num_inputs=obs_size, num_outputs=ac_size, hidden_dim=hidden_dim, 
                                            hidden_depth=hidden_depth, num_buckets=num_buckets, 
                                            ac_low=flattened_expert['actions'].min(axis=0) - ac_margin, 
                                            ac_high=flattened_expert['actions'].max(axis=0) + ac_margin)
        policy.to(device)
    elif args.policy == "diffusion":
        policy = DiffusionPolicy.DiffusionPolicy(obs_size=obs_size, obs_horizon=4, action_size=ac_size, action_horizon=8, action_pred_horizon=12, num_diffusion_iters=100, device=device)
    

    # Training hyperparameters
    if args.env == 'reacher':
        episode_length = 50
        num_epochs = 500
        batch_size = 32
    elif args.env == 'pointmaze':
        episode_length = 300
        num_epochs = 10
        batch_size = 128
        if args.train == "diffusion":
            num_epochs = 1
    else:
        raise ValueError('Invalid environment')

    if not args.test:
        if args.train == 'behavior_cloning':
            # Train behavior cloning
            simulate_policy_bc(env, policy, expert_data, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size)
        elif args.train == 'dagger':
            if args.env == 'reacher':
                # Load interactive expert
                expert_policy = torch.load('data/reacher_expert_policy.pkl', map_location=torch.device(device), weights_only=False)
                print("Expert policy loaded")
                expert_policy.to(device)
                ptu.set_gpu_mode(True)
            elif args.env == 'pointmaze':
                expert_policy = WaypointController(env.maze)
            else:
                raise ValueError('Invalid environment')

            num_dagger_iters=10
            num_epochs = int(num_epochs/num_dagger_iters)
            num_trajs_per_dagger=10

            simulate_policy_dagger(env, policy, expert_data, expert_policy, num_epochs=num_epochs, episode_length=episode_length,
                            batch_size=batch_size, num_dagger_iters=num_dagger_iters, num_trajs_per_dagger=num_trajs_per_dagger)
        elif args.train == "diffusion" and args.env == "pointmaze":
            policy = DiffusionPolicy.train_diffusion_policy(policy, get_expert_data(file_path), num_epochs=num_epochs, batch_size=batch_size)
        else:
            raise ValueError('Invalid training method')
        torch.save(policy.state_dict(), f'{args.policy}_{args.env}_{args.train}_final.pth')
    else:
        policy.load_state_dict(torch.load(f'{args.policy}_{args.env}_{args.train}_final.pth', weights_only=True))
    evaluate(env, policy, args.train, num_validation_runs=100, episode_length=episode_length, render=args.render, env_name=args.env)