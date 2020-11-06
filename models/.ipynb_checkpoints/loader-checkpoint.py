import torch as th
from torch import nn

from NeuralShield.Utils import loader
import numpy as np
from typing import Callable


def get_pi(env_name, algo):
    return loader.get_original_policy(env_name, algo)


def get_actor_net(env_name, algo):
    pi = get_pi(env_name, algo)
    params = pi.get_parameters()

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

    actor_net = ActorNetwork(weights, biases, logstd)

    return actor_net


class ActorNetwork(nn.Module):
    def __init__(self, weights, biases, logstd=None):
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

    def forward(self, x):
        return self.actor(x)

    def predict(self, obs: np.ndarray, deterministic=True) -> np.ndarray:
        obs = obs.astype(np.float32)
        obs = th.from_numpy(obs)
        with th.no_grad():
            mean = self.forward(obs)

        if deterministic:
            return mean.cpu().numpy()
