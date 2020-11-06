import numpy as np
import ray

from NeuralShield.Utils import loader


def simulation(env, pi, actor_net, step_num, rollout_num, attack_fn, attack_freq, attack_kwargs):
    """
    Simulation
    @param env: environment
    @param pi: policy
    @param actor_net: pytorch network
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
                obs = attack_fn(obs, actor_net, **attack_kwargs)
            action, _ = pi.predict(obs)
            obs, r, d, info = env.step(action)
            if r <= -100:
                unsafe_count += 1
                r = 0
            reward_sum += 3000 * r
        rewards.append(reward_sum)

    return np.mean(rewards), np.std(rewards), unsafe_count


def parallelized_simulation(env_name, algo, actor_net, step_num, rollout_num, attack_fn, attack_freq, attack_kwargs,
                            thread_number=20):

    @ray.remote
    def _surrogate():
        env = loader.get_env(env_name, algo)
        pi = loader.get_original_policy(env_name, algo)
        return simulation(env, pi, actor_net, step_num, rollout_num//thread_number, attack_fn, attack_freq, attack_kwargs)

    tids = [_surrogate.remote() for _ in range(thread_number)]
    res = [ray.get(tid) for tid in tids]

    ret = np.mean(res, axis=0)
    ret[-1] *= thread_number

    return ret
