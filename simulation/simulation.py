import numpy as np
import ray

from NeuralShield.Utils import loader


def simulation_attack_obs(env, pi, pi_net, q_net, step_num, rollout_num, attack_fn, attack_freq, attack_kwargs):
    """
    Simulation
    @param env: environment
    @param pi: policy
    @param pi_net: policy network
    @param q_net: q value network for RS attack
    @param step_num: simulation step for one rollout
    @param rollout_num: rollout number
    @param attack_fn: attack function
    @param attack_freq: attack frequency
    @param attack_kwargs: attack function keyword parameters
    @return: mean of reward, std of reward, number of unsafe state found
    """
    rewards = []
    unsafe_count = 0

    for _ in range(rollout_num):
        obs = env.reset()
        reward_sum = 0
        for _ in range(step_num):
            if np.random.random_sample() < attack_freq:
                if q_net is None:
                    obs = attack_fn(obs, pi_net, **attack_kwargs)
                else:
                    obs = attack_fn(obs, pi_net, q_net, **attack_kwargs)
            action, _ = pi.predict(obs)
            if len(action.shape) > 1:
                action = action[0]
            obs, r, d, info = env.step(action)
            if r <= -100:
                unsafe_count += 1
                r = 0
            reward_sum += 3000 * r
        rewards.append(reward_sum)

    return np.mean(rewards), np.std(rewards), unsafe_count


def parallelized_simulation_attack_obs(env_name, algo, pi_net, q_net, step_num, rollout_num, attack_fn, attack_freq,
                                       attack_kwargs,
                                       thread_number=20):
    @ray.remote
    def _surrogate():
        env = loader.get_env(env_name, algo, reward_type=None)
        pi = loader.get_original_policy(env_name, algo)
        return simulation_attack_obs(env, pi, pi_net, q_net, step_num, rollout_num // thread_number, attack_fn,
                                     attack_freq,
                                     attack_kwargs)

    tids = [_surrogate.remote() for _ in range(thread_number)]
    res = [ray.get(tid) for tid in tids]

    ret = np.mean(res, axis=0)
    ret[-1] *= thread_number

    return ret


def simulation_attack_state(env, pi, pi_net, q_net, step_num, rollout_num, attack_fn, attack_freq, attack_kwargs):
    """
    Simulation
    @param env: environment
    @param pi: policy
    @param pi_net: policy network
    @param q_net: q value network for RS attack
    @param step_num: simulation step for one rollout
    @param rollout_num: rollout number
    @param attack_fn: attack function
    @param attack_freq: attack frequency
    @param attack_kwargs: attack function keyword parameters
    @return: mean of reward, std of reward, number of unsafe state found
    """
    rewards = []
    unsafe_count = 0

    for _ in range(rollout_num):
        obs = env.reset()
        state = env.unwrapped.state
        reward_sum = 0
        for i in range(step_num):
            if np.random.random_sample() < attack_freq:
                if q_net is None:
                    state_hat = attack_fn(state, pi_net, **attack_kwargs)
                else:
                    state_hat = attack_fn(state, pi_net, q_net, **attack_kwargs)
                obs = env.reset(step_index=i, x0=state_hat)
            action, _ = pi.predict(obs)
            if len(action.shape) > 1:
                action = action[0]
            obs, r, d, info = env.step(action)
            state = env.unwrapped.state
            if r <= -100:
                unsafe_count += 1
                r = 0
            reward_sum += 3000 * r
        rewards.append(reward_sum)

    return np.mean(rewards), np.std(rewards), unsafe_count


def parallelized_simulation_attack_state(env_name, algo, pi_net, q_net, step_num, rollout_num, attack_fn, attack_freq,
                                         attack_kwargs,
                                         thread_number=20):
    @ray.remote
    def _surrogate():
        env = loader.get_env(env_name, algo, reward_type=None)
        pi = loader.get_original_policy(env_name, algo)
        return simulation_attack_state(env, pi, pi_net, q_net, step_num, rollout_num // thread_number, attack_fn,
                                       attack_freq,
                                       attack_kwargs)

    tids = [_surrogate.remote() for _ in range(thread_number)]
    res = [ray.get(tid) for tid in tids]

    ret = np.mean(res, axis=0)
    ret[-1] *= thread_number

    return ret
