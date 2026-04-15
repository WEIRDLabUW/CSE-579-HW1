"""
Diffusion Policy implementation built from scratch.

Fill in five TODO blocks that implement the core DDPM mechanics:
the noise schedule, the forward (noising) process, the reverse (denoising)
step, the training objective, and the sampling loop.

Reference papers:
  - Ho et al., "Denoising Diffusion Probabilistic Models", NeurIPS 2020
  - Chi et al., "Diffusion Policy", RSS 2023
"""

import math
import collections
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from policy import (
    BaseDiffusionDataset,
    ConditionalUnet1D,
    Policy,
    create_sample_indices,
    get_data_stats,
    normalize_data,
    torchify_dict,
    unnormalize_data,
)


# =====================================================================
# Noise scheduler
# =====================================================================
class NoiseScheduler:
    """Discrete-time DDPM noise scheduler.

    Stores beta_t, alpha_t, and alpha_bar_t, and
    implements the forward (training) and reverse (sampling) DDPM steps.
    """

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_train_timesteps = num_train_timesteps
        self.beta_schedule = beta_schedule
        self.device = device

        # ========== TODO 1: build the beta and alpha tables ==========
        # We need three tensors of length T = num_train_timesteps:
        #   betas      : the per-step forward variances beta_t
        #   alphas     : 1 - betas
        #   alphas_bar : cumulative product of alphas, i.e. prod_{s<=t} alpha_s
        # These will be used to build a "linear" schedule: betas linearly spaced from beta_start to beta_end.

        # ========== TODO 1: end ==========

        # Move to the right device.
        self.betas = betas.to(device=device)
        self.alphas = alphas.to(device=device)
        self.alphas_bar = alphas_bar.to(device=device)

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward (noising) process closed form.

        Returns x_t given x_0, the noise epsilon, and a batch of integer
        timesteps in [0, T).

        x_0       : (B, T_h, A) clean samples
        noise     : (B, T_h, A) standard normal noise of the same shape
        timesteps : (B,) integer timesteps
        """
        # ========== TODO 2: forward closed-form sample ==========
        # The DDPM forward process q(x_t | x_0) is a closed-form Gaussian:
        #   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        # Index into self.alphas_bar with `timesteps` to get a (B,) tensor,
        # then add singleton dims so it broadcasts against x_0.

        # ========== TODO 2: end ==========
        return x_t

    @torch.no_grad()
    def step(
        self,
        noise_pred: torch.Tensor,
        t: int,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """One reverse DDPM step.

        Returns x_{t-1} given x_t and the predicted noise eps_theta(x_t, t).
        Uses the direct epsilon-parameterized mean from Ho et al. 2020 eq. 11
        with sigma_t^2 = beta_t.
        """
        # ========== TODO 3: reverse DDPM step ==========
        # The mean of p_theta(x_{t-1} | x_t) is (Ho et al. eq. 11):
        #
        #   mu_theta = (1 / sqrt(alpha_t)) *
        #       (x_t  -  (beta_t / sqrt(1 - alpha_bar_t)) * eps_pred)
        #
        # The variance is simply sigma_t^2 = beta_t.
        #
        # For t > 0, sample  x_{t-1} = mu_theta + sqrt(beta_t) * z,  z ~ N(0,I)
        # For t == 0, return mu_theta directly (no noise).

        # ========== TODO 3: end ==========
        return x_prev


# =====================================================================
# Cosine LR with linear warmup.
# =====================================================================
def cosine_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Dataset -- builds windowed (state, action) samples.
# =====================================================================
class DiffusionDataset(BaseDiffusionDataset):
    def __init__(
        self,
        data,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        stats=None,
    ):
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

        train_data = {
            "state": states,
            "action": actions,
        }

        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        stats = dict() if stats is None else stats
        normalized_train_data = dict()
        for key, value in train_data.items():
            stats[key] = get_data_stats(value)
            normalized_train_data[key] = normalize_data(value, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon


# =====================================================================
# Diffusion policy
# =====================================================================
class DiffusionPolicy(Policy):
    def __init__(
        self,
        obs_size: int,
        obs_horizon: int,
        action_size: int,
        action_pred_horizon: int,
        action_horizon: int,
        num_diffusion_iters: int = 100,
        beta_schedule: str = "linear",
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
        self.obs_horizon = obs_horizon
        self.action_size = action_size
        self.action_horizon = action_horizon
        self.action_pred_horizon = action_pred_horizon
        self.num_diffusion_iters = num_diffusion_iters

        self.net = ConditionalUnet1D(action_size, obs_size * obs_horizon).to(device)
        self.scheduler = NoiseScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule=beta_schedule,
            device=device,
        )

        self.obs_deque = collections.deque([], maxlen=self.obs_horizon)
        self.stats = None

    def set_stats(self, stats):
        self.stats = torchify_dict(stats, self.device)

    @torch.no_grad()
    def _process_obs(self, obs: np.ndarray) -> Dict[str, torch.Tensor]:
        obs = np.copy(obs)
        return {
            "state": torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        }

    def reset(self) -> None:
        self.obs_deque.clear()

    def add_obs(self, obs: np.ndarray) -> None:
        o = self._process_obs(obs)
        self.obs_deque.append(o)
        while len(self.obs_deque) < self.obs_horizon:
            self.obs_deque.append(o)

    def __call__(self, obs: Optional[np.ndarray] = None) -> Any:
        return self.get_action(obs.squeeze() if obs is not None else None)

    @torch.no_grad()
    def get_action(self, obs: Optional[np.ndarray] = None):
        """Sample an action chunk from the diffusion policy.

        Returns (action, info) where action is a (action_horizon, action_dim)
        tensor of unnormalized actions to execute next.
        """
        assert self.stats is not None, (
            "Must call set_stats(...) before get_action() so the policy knows "
            "how actions and states were normalized at training time."
        )
        if obs is not None:
            self.add_obs(obs)
        assert len(self.obs_deque) == self.obs_horizon

        states = torch.stack([x["state"] for x in self.obs_deque])

        # ========== TODO 5: sampling loop ==========
        # 1. Normalize the observation history with self.stats['state'] and
        #    flatten it to a (1, obs_horizon * obs_dim) conditioning tensor
        #    that the UNet expects via its `global_cond` argument.
        #
        # 2. Initialize a tensor of pure Gaussian noise with shape
        #    (1, action_pred_horizon, action_size). This is x_K, the most
        #    noised "action" we will progressively denoise.
        #
        # 3. Loop t = num_diffusion_iters - 1, ..., 0:
        #       a. Predict the noise: eps_pred = self.net(naction, t, obs_cond)
        #       b. Step:               naction = self.scheduler.step(
        #                                            eps_pred, t, naction)
        #
        # 4. Drop the batch dimension and unnormalize with
        #    self.stats['action'] to recover real-valued actions.
        #
        # 5. Slice out the action_horizon actions starting from index
        #    obs_horizon - 1 -- this is the "action chunk" we will execute
        #    before re-querying the policy.

        # ========== TODO 5: end ==========
        return action, {}

    def state_dict(self):
        return dict(net=self.net.state_dict(), stats=self.stats)

    def load_state_dict(self, state_dict) -> None:
        self.net.load_state_dict(state_dict["net"])
        self.set_stats(state_dict["stats"])


# =====================================================================
# Training entry point.
#
# Boilerplate (dataset, dataloader, optimizer, LR schedule, logging)
# is provided. The TODO is the inner training step.
# =====================================================================
def train_diffusion_policy(
    policy: DiffusionPolicy,
    expert_data,
    num_epochs: int = 500,
    batch_size: int = 32,
):
    dataset = DiffusionDataset(
        expert_data,
        pred_horizon=policy.action_pred_horizon,
        obs_horizon=policy.obs_horizon,
        action_horizon=policy.action_horizon,
    )
    policy.set_stats(dataset.stats)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(
        policy.net.parameters(), lr=1e-4, weight_decay=1e-6
    )
    num_training_steps = len(data_loader) * num_epochs
    lr_scheduler = cosine_with_warmup(
        optimizer,
        num_warmup_steps=num_training_steps // 10,
        num_training_steps=num_training_steps,
    )
    losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        with tqdm(data_loader, leave=False) as tepoch:
            for batch in tepoch:
                naction = batch["action"].to(policy.device)
                nstate = batch["state"][:, : policy.obs_horizon].to(policy.device)
                B = nstate.shape[0]

                # ========== TODO 4: training step ==========
                # 1. Flatten the observation history into the global
                #    conditioning vector the UNet expects:
                #        obs_cond shape (B, obs_horizon * obs_dim)
                #
                # 2. Sample standard normal noise of the same shape as
                #    naction.
                #
                # 3. Sample one diffusion timestep per element of the batch:
                #        timesteps ~ Uniform{0, ..., T-1}, shape (B,)
                #
                # 4. Build the noisy actions x_t with the scheduler:
                #        noisy = policy.scheduler.add_noise(
                #            naction, noise, timesteps)
                #
                # 5. Predict the noise residual:
                #        noise_pred = policy.net(noisy, timesteps, obs_cond)
                #
                # 6. Compute MSE loss between predicted and true noise.

                # ========== TODO 4: end ==========

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

                loss_cpu = loss.item()
                running_loss += loss_cpu
                tepoch.set_postfix(loss=loss_cpu)
        epoch_loss = running_loss / len(data_loader)
        losses.append(epoch_loss)
        print(f"epoch {epoch} loss: {epoch_loss:.6f}")

    return policy
