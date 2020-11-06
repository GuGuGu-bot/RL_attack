import torch as th
from torch import nn

from NeuralShield.Utils import loader
import numpy as np
from typing import Callable

import NeuralShield
import os
ROOT = os.path.dirname(os.path.abspath(NeuralShield.__file__))

import pickle


def get_pi(env_name, algo):
    return loader.get_original_policy(env_name, algo)


def get_actor_net(env_name, algo, clip_obs=10):
    pi = get_pi(env_name, algo)
    params = pi.get_parameters()

    mu, std = None, None
    if "norm" in algo.split("_"):
        with open(f"{ROOT}/PretrainedModel/zoo/{algo.split('_')[0]}/{env_name}/obs_rms.pkl", "rb") as f:
            obs_rms = pickle.load(f)
            mu = obs_rms.mean.astype(np.float32)
            std = obs_rms.var.astype(np.float32)

    weights = []
    biases = []
    logstd = None
    for k in params:
        scopes = k.split("/")
        if "pi" in scopes[-2]:
            if "w" in scopes[-1]:
                weights.append(params[k])
            if "b" in scopes[-1]:
                biases.append(params[k])
            if "logstd" in scopes[-1]:
                logstd = params[k]

    actor_net = ActorNetwork(weights, biases, logstd, mu, std, clip_obs)

    return actor_net


class ActorNetwork(nn.Module):
    def __init__(self, weights, biases, logstd=None, mu=None, std=None, clip_obs=10):
        super(ActorNetwork, self).__init__()

        layers = []
        for w, b in zip(weights, biases):
            layer = nn.Linear(*w.shape)
            layer.weight.data = th.tensor(w.T, dtype=layer.weight.data.dtype)
            layer.bias.data = th.tensor(b, dtype=layer.bias.data.dtype)
            layers.append(layer)
            layers.append(nn.Tanh())

        layers.pop()

        self.actor = nn.Sequential(*layers)
        # TODO: add non-deterministic
        self.logstd = th.tensor(logstd, dtype=layers[-1].bias.data.dtype)

        self.mu = mu
        self.std = std
        self.clip_obs = clip_obs

    def forward(self, x):
        if self.mu is not None:
            assert self.std is not None
            x = (x - th.from_numpy(self.mu)) / th.from_numpy(self.std)
        return self.actor(x)

    def predict(self, obs: np.ndarray, deterministic=True) -> np.ndarray:
        obs = obs.astype(np.float32)
        obs = th.from_numpy(obs)
        with th.no_grad():
            mean = self.forward(obs)

        if deterministic:
            return mean.cpu().numpy()
