import torch
import numpy as np

from utils import rollout

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(env, policy, agent_name, num_validation_runs=10, episode_length=50, render=False, env_name=None):
    success_count = 0
    rewards_suc = 0
    rewards_all = 0
    for k in range(num_validation_runs):
        o, _ = env.reset()
        if agent_name == 'diffusion':
            policy.reset()
        path = rollout(
                env,
                policy,
                agent_name=agent_name,
                episode_length=episode_length,
                render=render)
        if agent_name == 'pg':
            success = len(path['dones']) == episode_length
        elif agent_name == 'behavior_cloning' or agent_name == "diffusion" or agent_name == 'dagger':
            if env_name == 'reacher':
                success = np.linalg.norm(env.unwrapped.get_body_com("fingertip") - env.unwrapped.get_body_com("target"))<0.1
            elif env_name == 'pointmaze':
                success = sum(path['dones']) > 0
            else:
                raise ValueError('Invalid environment')
        else:
            raise ValueError('Invalid agent name')
        if success:
            success_count += 1
            rewards_suc += np.sum(path['rewards'])
        rewards_all += np.sum(path['rewards'])
        print(f"test {k}, success {success}, reward {np.sum(path['rewards'])}")
    print("Success rate: ", success_count/num_validation_runs)
    print("Average reward (success only): ", rewards_suc/max(success_count, 1))
    print("Average reward (all): ", rewards_all/num_validation_runs)