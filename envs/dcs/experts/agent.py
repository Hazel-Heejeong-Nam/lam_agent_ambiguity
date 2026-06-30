import numpy as np
import torch
import torch.nn as nn
from gymnasium.wrappers.utils import update_mean_var_count_from_moments
from torch.distributions.normal import Normal


class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4, dtype=torch.float32):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape, dtype=dtype))
        self.register_buffer("var", torch.zeros(shape, dtype=dtype))
        self.register_buffer("count", torch.as_tensor(epsilon, dtype=dtype))

    def update(self, values):
        self._update_from_moments(
            torch.mean(values, dim=0),
            torch.var(values, dim=0),
            values.shape[0],
        )

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean,
            self.var,
            self.count,
            batch_mean,
            batch_var,
            batch_count,
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, hidden_dim=64, obs_norm_eps=1e-8):
        super().__init__()
        input_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = np.prod(envs.single_action_space.shape)

        self.obs_rms = RunningMeanStd(input_dim)
        self.obs_norm_eps = obs_norm_eps
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(input_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def normalize_obs(self, observations):
        normalized = (observations - self.obs_rms.mean) / torch.sqrt(
            self.obs_rms.var + self.obs_norm_eps
        )
        return torch.clamp(normalized, -10, 10)

    def get_action_and_value(self, observations, action=None, greedy=False):
        normalized = self.normalize_obs(observations)
        value = self.critic(normalized)
        action_mean = self.actor_mean(normalized)
        action_std = torch.exp(self.actor_logstd.expand_as(action_mean))
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = action_mean.detach() if greedy else distribution.sample()
        return (
            action,
            distribution.log_prob(action).sum(1),
            distribution.entropy().sum(1),
            value,
        )
