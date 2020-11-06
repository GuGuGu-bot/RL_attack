import torch as th
from typing import Callable
import numpy as np


def pgd(s_0: th.Tensor,
        loss_fn: Callable,
        delta: float = 1e-5,
        l_inf_norm: float = 0.0075,
        lr: float = 1e-3,
        iter_num: int = 10) -> np.ndarray:
    """
    PGD attack
    @param s_0: initial state
    @param loss_fn: attack will minimize this function
    @param delta: the initial delta, for loss == 0
    @param l_inf_norm: infinite norm boundary
    @param lr: learning rate of adam optimizer
    @param iter_num: iteration for optimizing
    @return: the attacked state
    """

    s_hat = th.nn.Parameter(s_0 + delta, requires_grad=True)
    optimizer = th.optim.Adam([s_hat], lr=lr)

    for _ in range(iter_num):
        optimizer.zero_grad()
        loss = loss_fn(s_0, s_hat)
        loss.backward()
        optimizer.step()
        s_hat.data = th.min(s_0 + l_inf_norm, th.max(s_0 - l_inf_norm, s_hat.data))

    return s_hat.data.cpu().detach().numpy()


def fgsm(s_0: th.Tensor,
         loss_fn: Callable,
         delta: float = 1e-5,
         l_inf_norm: float = 0.0075) -> np.ndarray:
    """
    FGSM
    @param s_0: initial state
    @param loss_fn: attack will minimize this function
    @param delta: the initial delta, for loss == 0
    @param l_inf_norm: infinite norm boundary
    @return: the attacked state
    """
    s_hat = th.nn.Parameter(s_0 + delta, requires_grad=True)
    loss = loss_fn(s_0, s_hat)
    loss.backward()
    print(np.sign(s_hat.grad.detach().cpu().numpy()))

    return s_0 + np.sign(s_hat.grad.detach().cpu().numpy()) * l_inf_norm


def mad_pgd(s_0: np.ndarray,
            pi_net,
            delta: float = 1e-5,
            l_inf_norm: float = 0.0075,
            lr: float = 1e-3,
            iter_num: int = 10) -> np.ndarray:
    """
    MAD PGD
    @param s_0: initial state
    @param pi_net: policy network
    @param delta: the initial delta, for loss == 0
    @param l_inf_norm: infinite norm boundary
    @param lr: learning rate of adam optimizer
    @param iter_num: iteration for optimizing
    @return: the attacked state
    """
    if pi_net.logstd is not None:
        sigma_inv = 1 / th.exp(pi_net.logstd)

    def loss_fn(s_0, s_hat):
        diff = pi_net(s_0) - pi_net(s_hat)
        return -th.sum(diff * sigma_inv * diff)

    s_0 = th.from_numpy(s_0.astype(np.float32))
    s_hat = pgd(s_0, loss_fn, delta, l_inf_norm, lr, iter_num)

    return s_hat


def mad_fgsm(s_0: np.ndarray,
             pi_net,
             delta: float = 1e-5,
             l_inf_norm: float = 0.0075):
    if pi_net.logstd is not None:
        sigma_inv = 1 / th.exp(pi_net.logstd)

    def loss_fn(s_0, s_hat):
        diff = pi_net(s_0) - pi_net(s_hat)
        return -th.sum(diff * sigma_inv * diff)

    s_0 = th.from_numpy(s_0.astype(np.float32))
    s_hat = fgsm(s_0, loss_fn, delta, l_inf_norm)

    return s_hat
