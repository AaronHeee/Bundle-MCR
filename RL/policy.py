from typing import Callable, Dict, List, Optional, Tuple, Type, Union, Any

import gym
import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
import random

from stable_baselines3.common.preprocessing import preprocess_obs, get_flattened_obs_dim
from stable_baselines3.common.utils import obs_as_tensor


item_embed = None 
MAX_VAL = 1e9

class OurCategorical(Categorical):
    def sample(self, sample_shape=th.Size()):
        if not isinstance(sample_shape, th.Size):
            sample_shape = th.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = th.multinomial(probs_2d, sample_shape.numel(), False).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

# Custom Policy Network

class ManagerPolicy(nn.Module):

    def __init__(
        self,
        lr_schedule = None,
        log_std_init: float = 0.0,
        use_sde: bool = False,
        use_expln: bool = False,
        squash_output: bool = False,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        args = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        super(ManagerPolicy, self).__init__()
        
        # arguments
        self.log_std_init = log_std_init
        self.args = args

        # model
        self._build()

        # optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def initialize_optimizer(self):
        self.optimizer = self.optimizer_class(self.parameters(), lr=5e-3, **self.optimizer_kwargs)

    def _build(self):

        # state proj
        self.proj = nn.Linear(7 * self.args.max_run, 64)

        # value network
        self.value_net =  nn.Sequential(
            nn.Linear(32+64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # policy network
        self.policy_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.policy_net_2 = None

        self.weight = nn.Sequential(
           nn.Linear(32 + 64, 32),
           nn.ReLU(),
           nn.Linear(32, 1),
           nn.Sigmoid() 
        )

    def reset_noise(self, n_envs: int = 1) -> None:
        pass

    def _extract_feature(self, obs, res_only=True):
        # record embeddings
        res = obs['result']
        res = th.cat([
                F.one_hot(r.long(), num_classes=7).float() for r in th.split(res.long(), 1, dim=1)
            ],dim=-1,
        ).view(res.shape[0], -1)
        res = self.proj(res)

        # slot embeddings
        slot = obs['slot']
        return res, slot

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        res, slot = self._extract_feature(obs)

        # Evaluate the values for the given observatre s
        prob_1 = F.softmax(self.policy_net(res), dim=-1)
        res = th.cat([res / res.norm(dim=-1, keepdim=True), slot / slot.norm(dim=-1, keepdim=True)], dim=-1)
        values = self.value_net(res)
        # values = self.value_net(slot)
        prob_2 = F.softmax(self.policy_net_2(slot), dim=-1)
        weight = self.weight(res)
        prob = weight * prob_1 + (1-weight) * prob_2
        distribution = Categorical(probs=prob)
        if deterministic:
            actions = th.argmax(distribution.probs, dim=1)
        else:
            actions = distribution.sample()
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        res, slot = self._extract_feature(obs)

        # values = self.value_net(th.cat([res, slot], dim=-1))
        # values = self.value_net(slot)
        # res = th.cat([res, slot], dim=-1)
        prob_1 = F.softmax(self.policy_net(res), dim=-1)
        res = th.cat([res / res.norm(dim=-1, keepdim=True), slot / slot.norm(dim=-1, keepdim=True)], dim=-1)
        values = self.value_net(res)
        # values = self.value_net(slot)
        prob_2 = F.softmax(self.policy_net_2(slot), dim=-1)
        weight = self.weight(res)
        prob = weight * prob_1 + (1-weight) * prob_2
        distribution = Categorical(probs=prob)
        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:

        self.set_training_mode(False)
        with th.no_grad():

            obs = obs_as_tensor(observation, self.device)
            res, slot = self._extract_feature(obs)
            # res = th.cat([res, slot], dim=-1)
            prob_1 = F.softmax(self.policy_net(res), dim=-1)
            prob_2 = F.softmax(self.policy_net_2(slot), dim=-1)
            res = th.cat([res / res.norm(dim=-1, keepdim=True), slot / slot.norm(dim=-1, keepdim=True)], dim=-1)
            weight = self.weight(res)
            prob = weight * prob_1 + (1-weight) * prob_2
            distribution = Categorical(probs=prob)
            if deterministic:
                actions = th.argmax(distribution.probs, dim=1)
            else:
                actions = distribution.sample()

        actions = actions.cpu().numpy()
        return actions, state

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)


class SidePolicy(nn.Module):

    def __init__(
        self,
        log_std_init: float = 0.0,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.SGD,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        embed_size = None
    ):

        super(SidePolicy, self).__init__()
        
        # arguments
        self.embed_size = embed_size
        self.log_std_init = log_std_init
        self.name = 'tag'

        # model
        self._build()

        # optimizer
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

    def initialize_optimizer(self):
        self.optimizer = self.optimizer_class(self.parameters(), lr=1e-3, weight_decay=0, **self.optimizer_kwargs)

    def _build(self):

        # shared network
        self.shared_net = nn.Sequential(
            nn.Flatten(),
        )
        
        # value network
        self.value_net =  nn.Sequential(
            nn.Linear(self.embed_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # policy network
        self.policy_net = None

    def reset_noise(self, n_envs: int = 1) -> None:
        pass

    def _extract_feature(self, obs):
        # results
        res = obs['embed']
        return res

    def forward(self, obs: th.Tensor, deterministic: bool = True, non_cand = None, k: int = 1) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        obs = self._extract_feature(obs)

        latent = self.shared_net(obs)
        values = self.value_net(latent)
        logits = self.policy_net(latent)

        distribution = OurCategorical(logits=logits[..., :-1])
        if deterministic:
            _, actions = th.topk(distribution.probs, k)
        else:
            actions = distribution.sample(th.Size([k]))
        actions = actions.flatten()
        log_prob = distribution.log_prob(actions).flatten()

        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:

        obs = self._extract_feature(obs)

        latent = self.shared_net(obs)
        values = self.value_net(latent)
        logits = self.policy_net(latent)

        distribution = OurCategorical(logits=logits[..., :-1])

        log_prob = distribution.log_prob(actions)
        return values, log_prob, distribution.entropy()

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)