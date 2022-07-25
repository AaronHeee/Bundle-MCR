from typing import Any, Dict, Generator, List, Optional, Union
from collections import defaultdict

import numpy as np
import torch as th

from stable_baselines3.common.type_aliases import DictRolloutBufferSamples

class DictRolloutBuffer():
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        device: Union[th.device, str] = "cpu",
        gae_lambda: float = 1,
        gamma: float = 0.99,
    ):
        self.device = device
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.reset()

    def reset(self) -> None:
        self.observations = defaultdict(list)
        self.actions, self.rewards, self.advantages = [], [], []
        self.returns, self.episode_starts, self.values, self.log_probs = [], [], [], []
        self.generator_ready = False
        self.full = False
        self.pos = 0

    def add(self,records, episode_start):
        """
        :param records: 
            :key obs: dict, obs[i] is a cpu tensor, (1, *)
            :key reward: float
            :key action: cpu tensor, (1, )
            :key value: cpu tensor, (1, )
            :key log_prob: cpu tensor, (1, )
            :key episode_start: int (0 or 1)
        """

        lengths = []
        for key in records['obs'].keys():
            self.observations[key].append(records['obs'][key])
            lengths.append(len(self.observations[key]))

        self.actions.append(records['action'])
        lengths.append(len(self.actions))
        self.rewards.append(records['reward'])
        lengths.append(len(self.rewards))
        self.episode_starts.append(episode_start)
        lengths.append(len(self.episode_starts))
        self.values.append(records['value'])
        lengths.append(len(self.values))
        self.log_probs.append(records['log_prob'])
        lengths.append(len(self.log_probs))

        if len(set(lengths)) != 1:
            import pdb; pdb.set_trace()
        assert len(set(lengths)) == 1 # equal length
        
        self.pos += 1


    def get(self, batch_size: Optional[int] = None) -> Generator[DictRolloutBufferSamples, None, None]:
        total_size = len(self.advantages)
        indices = np.random.permutation(total_size)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = th.cat(obs, dim=0)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = th.cat(self.__dict__[tensor], dim=0)
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = total_size

        start_idx = 0
        while start_idx < total_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds):
        return DictRolloutBufferSamples(
            observations={key: obs[batch_inds].to(self.device) for (key, obs) in self.observations.items()},
            actions=self.actions[batch_inds].long().to(self.device),
            old_values=self.values[batch_inds].flatten().to(self.device),
            old_log_prob=self.log_probs[batch_inds].flatten().to(self.device),
            advantages=self.advantages[batch_inds].flatten().to(self.device),
            returns=self.returns[batch_inds].flatten().to(self.device),
        )

    def compute_returns_and_advantage(self) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain vanilla advantage (A(s) = R - V(S))
        where R is the discounted reward with value bootstrap,
        set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).

        """
        # Convert to numpy

        last_gae_lam = 0

        for step in reversed(range(self.pos-1)):
            next_non_terminal = 1.0 - self.episode_starts[step + 1]
            next_values = self.values[step + 1]
            #calculate using https://github.com/DLR-RM/stable-baselines3/pull/375. 
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages.append(last_gae_lam)

        self.advantages = self.advantages[::-1]
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = [a + v for a, v in zip(self.advantages, self.values[:-1])]
