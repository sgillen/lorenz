import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from multiprocessing import Pool
from stable_baselines.common import make_vec_env
import time
import seagul.envs.bullet
import seagul.envs.simple_nonlinear
from seagul.integration import euler
import os
import matplotlib.pyplot as plt
from seagul.plot import smooth_bounded_curve, chop_returns
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

num_steps = int(2e6)
base_dir = "./data/linear_z/ppo/"
trial_name = input("Trial name: ")

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def reward_fn(s):
    if s[3] > 0:
        if s[0] >= 0 and s[2] >= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[2]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2) - 5)**2,0,5)
            s[3] = -10
        else:
            reward = 0.0

    elif s[3] < 0:
        if s[0] <= 0 and s[2] <= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[2]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2)**2 - 5),0,5)
            s[3] = 10
        else:
            reward = 0.0

    return reward, s


def run_stable(seed):
    save_dir = trial_dir + "/" + str(seed)
    os.makedirs(save_dir, exist_ok=False)

    env_steps = 500
    env_config = {
        "reward_fn": reward_fn,
        "xyz_max": float('inf'),
        "num_steps": env_steps,
        "act_hold": 10,
        "integrator": euler,
        "dt": .01,
        "init_noise_max": 10,
    }

    env = make_vec_env(seagul.envs.simple_nonlinear.LinearEnv, n_envs=4, monitor_dir=save_dir, env_kwargs=env_config)
    model = PPO2(MlpPolicy,
                 env,
                 verbose=2,
                 seed=int(seed),
                 # normalize= True,
                 # policy= 'MlpPolicy',
                 n_steps=1024,
                 nminibatches=64,
                 lam=0.95,
                 gamma=0.99,
                 noptepochs=10,
                 ent_coef=0.0,
                 learning_rate=2.5e-4,
                 cliprange=0.1,
                 cliprange_vf=-1,
                 )

    model.learn(total_timesteps=num_steps)
    model.save(save_dir + "/model.zip")


if __name__ == "__main__":

    start = time.time()

    seeds = np.random.randint(0, 2 ** 32, 8)
    with Pool(processes=8) as pool:
        pool.map(run_stable, seeds)

    #results = chop_returns(results)
    #results = np.array(results).transpose(1, 0)

    #smooth_bounded_curve(results)
    #plt.show()

    print(f"experiment complete, total time: {time.time() - start}, saved in {trial_dir}")

