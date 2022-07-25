from email.policy import default
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import os
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from collections import defaultdict

from .buffer import DictRolloutBuffer
from .utils import metrics

from stable_baselines3.common.utils import obs_as_tensor, get_schedule_fn

class HierPPO():
    def __init__(self, env, args):
        self.env = env
        self.args = args
        self._setup_model()

    def _setup_model(self) -> None:

        # conv policy and buffer
        self.policy = self.env.conv_policy
        self.buffer = DictRolloutBuffer(
            self.args.device,
            gamma=self.args.gamma,
            gae_lambda=self.args.gae_lambda,
        )

        # side policy and buffer
        self.side_policies = self.env.side_policies
        self.side_buffers = {
            n: DictRolloutBuffer(self.args.device, gamma=self.args.gamma, gae_lambda=self.args.gae_lambda) 
            for n in self.side_policies
        }

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.args.clip_range)

    def collect_rollouts(self, env, n_rollout_steps):
        n_steps = 0
        self.buffer.reset()
        self._last_obs = self.env.reset() 
        self._last_episode_starts = 1
        metric_records = defaultdict(list)

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        for n in self.side_policies:
            self.side_buffers[n].reset()
            self.side_policies[n].set_training_mode(False)
        
        tqdm_loader = tqdm(range(n_rollout_steps))
        for i in tqdm_loader:

            # inference
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.args.device)
                action, value, log_prob = self.policy.forward(obs_tensor)

            # interaction
            action = action.item()
            new_obs, conv_record, side_records = env.step(action, self._last_obs, value.squeeze(-1).cpu(), log_prob.cpu()) # transfer to rewards
            self.num_timesteps += 1
            n_steps += 1

            # record
            self.buffer.add(conv_record, self._last_episode_starts)
            
            # add rollout buffer for item / category / attribute prediction head
            for n in self.side_policies:
                for i, side_record in enumerate(side_records[n]):
                    self.side_buffers[n].add(side_record, self._last_episode_starts if i ==0 else 0)

            # update
            self._last_episode_starts = conv_record['done']
            self._last_obs = new_obs
            if conv_record['done']:
                # start a new episode and record results
                for m in metrics:
                    metric_records[m].append(metrics[m](self.env.target_set, self.env.partial_list, self.env.all_set))
                metric_records['run'].append(self.env.cur_conver_step)
                metric_records['reward'].append(conv_record['reward'])
                tqdm_loader.set_description(f"[{self.num_timesteps}/{self.total_timesteps}] " + ", ".join([f"{m}:{np.mean(metric_records[m]): .4f}" for m in metric_records]))
                self._last_obs = env.reset()

        self.buffer.compute_returns_and_advantage()
        for n in self.side_policies:
            self.side_buffers[n].compute_returns_and_advantage()

    def train_one_epoch(self, policy, rollout_buffer, logs, params, epoch, name=None):
        # Switch to train mode (this affects batch norm / dropout)
        policy.set_training_mode(True)
        approx_kl_divs = []
        # Do a complete pass on the rollout buffer
        for rollout_data in rollout_buffer.get(self.args.batch_size):
            actions = rollout_data.actions

            values, log_prob, entropy = policy.evaluate_actions(rollout_data.observations, actions)
            values = values.flatten()

            # Normalize advantage
            advantages = rollout_data.advantages 
            # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ratio between old and new policy, should be one at the first iteration
            ratio = torch.exp(log_prob - rollout_data.old_log_prob) 

            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(ratio, 1 - params['clip_range'], 1 + params['clip_range'])
            policy_loss = - torch.min(policy_loss_1, policy_loss_2).mean()

            # Logging
            logs['pg_losses'].append(policy_loss.item())
            clip_fraction = torch.mean((torch.abs(ratio - 1) > params['clip_range']).float()).item()
            logs['clip_fractions'].append(clip_fraction)

            if self.clip_range_vf is None:
                # No clipping
                values_pred = values
            else:
                # Clip the different between old and new value
                # NOTE: this depends on the reward scaling
                values_pred = rollout_data.old_values + torch.clamp(
                    values - rollout_data.old_values, -params['clip_range_vf'], params['clip_range_vf']
                )
            # Value loss using the TD(gae_lambda) target
            value_loss = F.mse_loss(rollout_data.returns, values_pred)
            logs['value_losses'].append(value_loss.item())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -torch.mean(-log_prob)
            else:
                entropy_loss = -torch.mean(entropy)

            loss = policy_loss + self.args.vf_coef * value_loss
            logs['losses'].append(loss.item())

            # Optimization step
            policy.optimizer.zero_grad()
            loss.backward()
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(policy.parameters(), self.args.max_grad_norm)
            policy.optimizer.step()

        logs['approx_kl_divs'].append(approx_kl_divs)
        return policy, logs


    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        conv_logs = {
            'pg_losses': [], 
            'value_losses': [],
            'clip_fractions': [],
            'approx_kl_divs': [],
            'losses': []
        }
        side_logs = {
            n: {
                'pg_losses': [], 
                'value_losses': [],
                'clip_fractions': [],
                'approx_kl_divs': [],
                'losses': [],
            } for n in self.side_policies
        }
        params = {
            'clip_range': clip_range,
            'clip_range_vf': clip_range_vf if self.clip_range_vf is not None else None
        }

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.args.n_epochs):
            if self.num_timesteps > 0: 
                self.policy, conv_logs = self.train_one_epoch(self.policy, self.buffer, conv_logs, params, epoch, name='manager_policy')
                for n in self.side_policies:
                    self.side_policies[n], side_logs[n] = self.train_one_epoch(self.side_policies[n], self.side_buffers[n], side_logs[n], params, epoch, name=n)
            if not continue_training:
                break

    def learn(
        self,
        total_timesteps,
        n_eval_episodes=1,
        n_eval_samples=512,
        eval_log_path=None,
    ):
        iteration = 0
        self.num_timesteps = 0
        self.start_time = time.time()
        self.clip_range_vf = None
        self.total_timesteps = total_timesteps
        best_reward = 0

        while self.num_timesteps < self.total_timesteps:

            # collect
            self.collect_rollouts(self.env, n_rollout_steps=self.args.n_steps)

            # display collected infos
            iteration += 1
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(self.total_timesteps)

            if n_eval_episodes is not None and iteration % n_eval_episodes == 0:
                res = self.env.evaluate(self, n_samples=n_eval_samples)
                if res['rewards'] >= best_reward:
                    best_reward = res['rewards']
                    print("[!] saving model...")
                    torch.save(self.policy.state_dict(), os.path.join(self.args.ckpt_dir, "conv_manager.pt"))
                    torch.save(self.env.bundle_rec.state_dict(), os.path.join(self.args.ckpt_dir, "bunt.pt"))

            # training
            self.train()

        return self