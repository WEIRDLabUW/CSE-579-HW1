import pickle
import numpy as np
from diffusers import DDPMScheduler, get_scheduler, EMAModel
from policy import torchify_dict, ConditionalUnet1D, Policy, BaseDiffusionDataset, create_sample_indices, get_data_stats, normalize_data, unnormalize_data
import torch
from typing import Any, Dict, Optional
import collections
from tqdm.auto import tqdm

class DiffusionDataset(BaseDiffusionDataset):
    def __init__(self, data, pred_horizon: int, obs_horizon: int, action_horizon: int, stats=None):
        actions = []
        states = []
        episode_ends = []

        for trajectory in data:
            state = np.array(trajectory["observations"])
            states.append(state)
            actions.append(np.array(trajectory["actions"]))
            if len(episode_ends) == 0:
                episode_ends.append(len(state))
            else:
                episode_ends.append(episode_ends[-1] + len(state))
        actions = np.concatenate(actions).astype(np.float32)
        states = np.concatenate(states).astype(np.float32)
        episode_ends = np.array(episode_ends)

        # (N, D)
        train_data = {
            # first two dims of state vector are agent (i.e. gripper) locations
            'state': states,
            'action': actions,
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1)

        # compute statistics and normalized data to [-1,1]
        stats = dict() if stats is None else stats
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)#  if key not in stats else stats[key]
            normalized_train_data[key] = normalize_data(data, stats[key])
        print(stats)
        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


class DiffusionPolicy(Policy):
    def __init__(self, obs_size: int, obs_horizon: int, action_size: int, action_pred_horizon: int , action_horizon: int, num_diffusion_iters=100, device: torch.device=torch.device('cuda')):

        self.net = ConditionalUnet1D(action_size, obs_size * obs_horizon ).to(device) #down_dims=[32, 64, 128]
        self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=num_diffusion_iters,
                # the choise of beta schedule has big impact on performance
                # we found squared cosine works the best
                beta_schedule='squaredcos_cap_v2',
                # clip output to [-1,1] to improve stability
                clip_sample=True,
                # our network predicts noise (instead of denoised action)
                prediction_type='epsilon'
            )

        self.obs_horizon = obs_horizon
        self.device = device
        self.action_size = action_size
        self.obs_deque = collections.deque([], maxlen=self.obs_horizon)
        self.num_diffusion_iters = num_diffusion_iters
        self.action_horizon = action_horizon
        self.action_pred_horizon = action_pred_horizon
        # need to be set later
        self.stats = None
        
    def set_stats(self, stats):
        self.stats = torchify_dict(stats, self.device)

    @torch.no_grad()
    def _process_obs(self, obs:  np.ndarray) -> Dict[str, torch.Tensor]:
        ret = {}
        obs = np.copy(obs)
        ret["state"] = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device)
        return ret

    def reset(self) -> None:
        self.obs_deque.clear()

    def add_obs(self, obs: np.ndarray) -> None:
        o = self._process_obs(obs)
        self.obs_deque.append(o)
        while len(self.obs_deque) < self.obs_horizon:
            self.obs_deque.append(o)

    def __call__(self,  obs: Optional[np.ndarray] = None) -> Any:
        return self.get_action(obs.squeeze() if obs is not None else None)
    
    @torch.no_grad()
    def get_action(self, obs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Takes dict with "embed" and "state" possibly as keys, and returns actions.
        actions is a (H, D) array, where H is the action horizon and D is action dimension.
        """
        assert self.stats, "Need to call set_data_stats before calling get_action which are the normalization paramaters"
        if obs is not None:
            self.add_obs(obs)
        assert len(self.obs_deque) == self.obs_horizon
        
        states = torch.stack([x["state"] for x in self.obs_deque])

        #========== TODO: start ==========
        # normalize the states to feed into the policy. For this use the self.stats dictionary and
        # the normalize_data function.

        # reshape the states to be the correct shape to pass into the policy network.

        # initialize action from Gaussian noise
        naction = np.zeros(1) # TODO fill this in

        #========== TODO: end ==========
        # init the DDPM scheduler
        self.noise_scheduler.set_timesteps(
            self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            #========== TODO: start ==========
            # Run one single iterative denoising step on the naction.abs
            # Use the self.net to predict the noise based on the timestep k

            # run the inverse diffusion step using the noise_scheduler.step function. See https://huggingface.co/docs/diffusers/en/api/schedulers/ddpm
            # for more information on the noise_scheduler.step function.
            pass
            #========== TODO: end ==========
        
        # normalized action ouptut should be batchsize, pred_horizon, action_dimention
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        
        # unnormalize the action
        action_pred: torch.Tensor = unnormalize_data(
            naction, stats=self.stats['action'])

        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = action_pred[start:end, :]

        return action, {}
    
    def state_dict(self):
        return dict(net=self.net.state_dict(),
                    stats=self.stats )
    
    def load_state_dict(self, state_dict) -> None:
        self.net.load_state_dict(state_dict["net"])
        self.set_stats(state_dict["stats"])
    
    
def train_diffusion_policy(policy: DiffusionPolicy, expert_data, num_epochs=500, batch_size=32):
    # Diffusion Training Function
    dataset = DiffusionDataset(expert_data, pred_horizon=policy.action_pred_horizon, obs_horizon=policy.obs_horizon, action_horizon=policy.action_horizon)
    policy.set_stats(dataset.stats)
    print(policy.action_horizon, policy.obs_horizon, policy.action_pred_horizon)
    ema = EMAModel(
        parameters=policy.net.parameters(),
        power=0.75)
    print(dataset.stats)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    optimizer = torch.optim.AdamW(policy.net.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=(len(data_loader) * num_epochs) // 10,
        num_training_steps=len(data_loader) * num_epochs
    )
    losses = []
    
    
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0 
        with tqdm(data_loader, leave=False) as tepoch:
            for batch in tepoch:
                # normalized action and state from batch
                naction = batch['action'].to(policy.device)
                nagent_pos = batch['state'][:, :policy.obs_horizon].to(policy.device)
                B = nagent_pos.shape[0]

                #========== TODO: begin ==========

                # first reshape the conditioning to fit into the policy by flattening it.
                # Then sample noise to add to the actions.

                # Code provided to sample a diffusion iteration for each data point. For more information about the noise
                # scheduler, see https://arxiv.org/pdf/2006.11239
                timesteps = torch.randint(
                    0, policy.noise_scheduler.config.num_train_timesteps,
                    (B,), device=policy.device
                ).long()

                # Use the policy.noise_scheduler to add noise to the normalized actions based on the sampled
                # noise and time steps

                # predict the noise residual using self.net

                # Calculate the loss between the predicted and actual noise

                #========== TODO: end ==========
                # optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(policy.net.parameters())

                # logging
                loss_cpu = loss.item()
                running_loss += loss_cpu
                tepoch.set_postfix(loss=loss_cpu)
        losses.append(running_loss / len(data_loader))
        print(running_loss/ len(data_loader))
    
    # set the final weights to the EMA
    ema_model = policy.net
    ema.copy_to(ema_model.parameters())
    policy.net = ema_model
    return policy
