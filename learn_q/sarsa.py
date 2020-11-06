import torch as th
from torch import nn
import numpy as np

from NeuralShield.Utils import loader


class QNetwork(nn.Module):
    def __init__(self, s_size, a_size):
        super(QNetwork, self).__init__()
        self.q_net = nn.Sequential(
            nn.Linear(s_size + a_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, s, a):
        return self.q_net(th.cat([s, a], dim=-1))


class SARSA:
    def __init__(self, env, actor, q_net: QNetwork, optimizer=None, attack_type="state"):
        assert env.reward_type == "safety"

        self.memory = []
        self.env = env  # with safety reward
        self.actor = actor
        self.q_net = q_net

        if optimizer is None:
            self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        self.attack_type = attack_type
        self.losses = []

    def _rollout(self, step_num):
        obs = self.env.reset()

        rollout = []
        if self.attack_type == "state":
            state = self.env.unwrapped.state

            for _ in range(step_num):
                if len(rollout) > 0:
                    rollout[-1].append(action)
                sarsa = [state]  # collect state instead observation
                action, _ = self.actor.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                state = self.env.unwrapped.state
                sarsa.extend([action, reward, state])
                rollout.append(sarsa)
                if reward <= -100:
                    rollout[-1][2] = -1
                    break
        else:
            for _ in range(step_num):
                if len(rollout) > 0:
                    rollout[-1].append(action)
                sarsa = [obs]
                action, _ = self.actor.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                sarsa.extend([action, reward, obs])
                rollout.append(sarsa)
                if reward <= -100:
                    rollout[-1][2] = -1
                    break

        self.memory.append(rollout)

    def update_network(self, iter_number, gamma):
        for _ in range(iter_number):
            loss = th.zeros(1)
            for traj in self.memory:
                traj = np.array(traj[:-1])
                s0 = th.from_numpy(np.array(traj[:, 0].tolist()).astype(np.float32))
                a0 = th.from_numpy(np.array(traj[:, 1].tolist()).astype(np.float32))
                r0 = th.from_numpy(np.array(traj[:, 2].tolist()).astype(np.float32))
                s1 = th.from_numpy(np.array(traj[:, 3].tolist()).astype(np.float32))
                a1 = th.from_numpy(np.array(traj[:, 4].tolist()).astype(np.float32))

                q_0 = self.q_net(s0, a0)
                q_1 = self.q_net(s1, a1)

                loss += th.mean(th.abs(q_0.squeeze() - (r0 + gamma * q_1.squeeze())))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.losses.append(loss.cpu().detach().numpy())

    def train(self, step_num, rollout_num, update_intv, update_iter, gamma=0.99):
        for i in range(rollout_num):
            self._rollout(step_num)
            if i % update_intv == 0 and i != 0:
                self.update_network(update_iter, gamma)
                self.memory = []

        return self.q_net


def learn_q(env_name,
            algo,
            step_num,
            rollout_num,
            update_intv,
            update_iter,
            gamma=0.99,
            attack_type="state",
            optimizer=None):
    env = loader.get_env(env_name, algo)
    actor = loader.get_original_policy(env_name, algo)
    if attack_type == "state":
        env.reset()
        s_size = len(env.unwrapped.state)
    else:
        s_size = env.observation_space.shape[0]
    a_size = env.action_space.shape[0]
    q_net = QNetwork(s_size, a_size)

    sarsa = SARSA(env, actor, q_net, optimizer, attack_type=attack_type)
    sarsa.train(step_num, rollout_num, update_intv, update_iter, gamma)

    return q_net, sarsa.losses


def _tc1():
    q_net = QNetwork(4, 2)
    s = th.zeros(4)
    a = th.zeros(2)
    print(q_net(s, a))

    s = th.zeros([3, 4])
    a = th.zeros([3, 2])
    print(q_net(s, a))


def _tc2():
    env = loader.get_env("AntBulletEnv-v0", "ppo2_norm")
    actor = loader.get_original_policy("AntBulletEnv-v0", "ppo2_norm")
    env.reset()
    s_size = len(env.unwrapped.state)
    a_size = env.action_space.shape[0]
    q_net = QNetwork(s_size, a_size)

    sarsa = SARSA(env, actor, q_net, attack_type="state")
    sarsa.train(1000, 100, 5, 10)


def _tc3():
    env = loader.get_env("AntBulletEnv-v0", "ppo2_norm")
    actor = loader.get_original_policy("AntBulletEnv-v0", "ppo2_norm")
    obs = env.reset()
    s_size = len(obs)
    a_size = env.action_space.shape[0]
    q_net = QNetwork(s_size, a_size)

    sarsa = SARSA(env, actor, q_net, attack_type="obs")
    sarsa.train(1000, 100, 5, 10)


if __name__ == '__main__':
    _tc3()
