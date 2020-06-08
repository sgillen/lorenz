from multiprocessing import Process
import torch.nn as nn
import numpy as np

# init policy, value fn

env_name = "linear_z2d-v0"

input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

from seagul.rl.run_utils import run_sg
from seagul.rl.ppo import ppo, PPOModel
from seagul.nn import MLP
from seagul.integration import euler
import seagul.envs

#from seagul.integrationx import euler

proc_list = []
trial_num = input("What trial is this?\n")


def reward_fn(s):
    if s[2] > 0:
        if 12 > s[0] > 2 and 13 > s[1] > 3:
            reward = 5.0
            s[2] = -10
        else:
            reward = 0.0

    elif s[2] < 0:
        if -12 < s[0] < -2 and -13 < s[1] < -3:
            reward = 5.0
            s[2] = 10
        else:
            reward = 0.0

    return reward, s


for seed in np.random.randint(0, 2 ** 32, 8):
    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)

    model = PPOModel(
        policy=policy,
        value_fn=value_fn,
        fixed_std=False
    )

    num_steps = 50
    env_config = {
        "reward_fn": reward_fn,
        "xz_max": float('inf'),
        "num_steps": num_steps,
        "act_hold": 10,
        "integrator": euler,
        "dt": .01,
        "init_noise_max": 10,
    }

    # def len_fn(rews):
    #     if len(rews) < 5:
    #         return 50
    #     else:
    #         return np.clip(sum(rews[-5:])/5*2, 50, 5000)

    len_schedule = np.asarray([50, 1000])
    sched_length = len_schedule.shape[0]
    x_vals = np.linspace(0, 2e6, sched_length)
    len_lookup = lambda steps: np.interp(steps, x_vals, len_schedule)

    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "total_steps": 2e6,
        "epoch_batch_size": 1024,
        "sgd_batch_size": 512,
        "lam": .2,
        "gamma": .95,
        "sgd_epochs" : 50,
        "normalize_return": True,
        "env_config": env_config,
    }

    # run_sg(alg_config, ppo, "ppo", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

    p = Process(
        target=run_sg,
        args=(alg_config, ppo, "ppo", "", "/data2/2d_hole/" + trial_num + "/" + "seed" + str(seed)),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
