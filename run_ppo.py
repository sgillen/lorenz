from multiprocessing import Process
import seagul.envs

import gym

env_name = "linear_z-v0"
import torch
import torch.nn as nn
import numpy as np
from numpy import pi

# init policy, value fn
input_size = 4
output_size = 2
layer_size = 64
num_layers = 4
activation = nn.ReLU

from seagul.rl.run_utils import run_sg, run_and_save_bs
from seagul.rl.ppo import ppo, PPOModel, PPOModelActHold
from seagul.nn import MLP, CategoricalMLP
from seagul.integration import euler, rk4
#from seagul.integrationx import euler


proc_list = []
trial_num = input("What trial is this?\n")


def reward_fn(s):
    if s[3] > 0:
        if 12 > s[0] > 2 and 13 > s[2] > 3:
            reward = 5.0
            s[3] = -10
        else:
            reward = 0.0

    elif s[3] < 0:
        if -12 < s[0] < -2 and -13 < s[2] < -3:
            reward = 5.0
            s[3] = 10
        else:
            reward = 0.0

    return reward, s


for var in [2]:
    for seed in np.random.randint(0, 2 ** 32, 8):
        policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation)

        model = PPOModel(
            policy=policy,
            value_fn=value_fn,
            fixed_std=False
        )

        env_config = {
            "reward_fn": reward_fn,
            "xyz_max": float('inf'),
            "num_steps": 100,
            "act_hold": 10,
            "integrator": euler,
            "dt": .01,
        }

        alg_config = {
            "env_name": env_name,
            "model": model,
            "act_var_schedule": [var],
            "seed": int(seed),  # int((time.time() % 1)*1e8),
            "total_steps": 2e6,
            "epoch_batch_size": 1024,
            "pol_batch_size": 512,
            "val_batch_size": 1024,
            "lam": .2,
            "gamma": .95,
            "normalize_return": False,
            "env_config": env_config
        }


        #run_sg(alg_config, ppo, "ppo", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

        p = Process(
            target=run_sg,
            args=(alg_config, ppo, "ppo", "longer_trials and sweep over variance", "/data2/sg_ppo_fixed/" + trial_num + "_" + str(var) + "/" + "seed" + str(seed)),
        )
        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()
