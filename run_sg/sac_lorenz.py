from multiprocessing import Process

env_name = "linear_z-v0"
import torch.nn as nn
import numpy as np

# init policy, value fn
input_size = 4
output_size = 2
layer_size = 32
num_layers = 2
activation = nn.ReLU

from seagul.rl.run_utils import run_sg
from seagul.rl.sac import sac, SACModel
from seagul.nn import MLP
from seagul.integration import euler
import seagul.envs
import gym
#from seagul.integrationx import euler


proc_list = []
trial_num = input("What trial is this?\n")

#
# def reward_fn(s):
#     if s[3] > 0:
#         if s[0] >= 0 and s[2] >= 0:
#             reward = s[0]
#             s[3] = -10
#         else:
#             reward = 0.0
#
#     elif s[3] < 0:
#         if s[0] <= 0 and s[2] <= 0:
#             reward = -s[0]
#             s[3] = 10
#         else:
#             reward = 0.0
#
#     return reward, s


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


for seed in np.random.randint(0, 2 ** 32, 8):

    num_steps = 500
    env_config = {
        "reward_fn": reward_fn,
        "xyz_max": float('inf'),
        "num_steps": num_steps,
        "act_hold": 10,
        "integrator": euler,
        "dt": .01,
        "init_noise_max": 10,
    }
    env = gym.make(env_name, **env_config)

    model = SACModel(
        policy=MLP(input_size, output_size * 2, num_layers, layer_size, activation),
        value_fn=MLP(input_size, 1, num_layers, layer_size, activation),
        q1_fn=MLP(input_size + output_size, 1, num_layers, layer_size, activation),
        q2_fn=MLP(input_size + output_size, 1, num_layers, layer_size, activation),
        act_limit=env.action_space.high[0]
    )

    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "total_steps": 4e5,
        "alpha": .05,
        "exploration_steps": 5000,
        "min_steps_per_update": 500,
        #"reward_stop": 950,
        "env_steps": 0,
        "gamma": .99,
        "sgd_batch_size": 128,
        "replay_batch_size": 128,
        "iters_per_update": float('inf'),
        "env_config" : env_config,
    }

    #run_sg(alg_config, sac, "sac", "sac with inverted pend params", "/data/sac/" + trial_num + "/" + "seed" + str(seed))


    p = Process(
        target=run_sg,
        args=(alg_config, sac, "sac", "sac with inverted pend params", "/data/sac/" + trial_num + "/" + "seed" + str(seed)),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
