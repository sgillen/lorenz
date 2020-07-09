from multiprocessing import Process

env_name = "linear_z2d-v0"
import torch.nn as nn
import numpy as np

# init policy, value fn
input_size = 3
output_size = 1
layer_size = 64
num_layers = 2
activation = nn.ReLU

from seagul.rl.run_utils import run_sg
from seagul.rl.ppo import ppo_visit, PPOModel
from seagul.nn import MLP
from seagul.integration import euler
import seagul.envs

#from seagul.integrationx import euler


proc_list = []
trial_num = input("What trial is this?\n")



def reward_fn(s):
    if s[2] > 0:
        if s[0] >= 0 and s[1] >= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2) - 5)**2,0,5)
            s[2] = -10
        else:
            reward = 0.0

    elif s[2] < 0:
        if s[0] <= 0 and s[1] <= 0:
            reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
            #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2)**2 - 5),0,5)
            s[2] = 10
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


        num_steps = 500
        env_config = {
            "reward_fn": reward_fn,
            "xz_max": float('inf'),
            "num_steps": num_steps,
            "act_hold": 10,
            "integrator": euler,
            "dt": .01,
            "init_noise_max": 10,
        }


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
            "env_config": env_config,
        }

        #        run_sg(alg_config, ppo_visit, "ppo", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

        p = Process(
            target=run_sg,
            args=(alg_config, ppo_visit, "ppo", " visit with shorter episode", "/data22/visit/2d" + trial_num  + "/" + "seed" + str(seed)),
        )
        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()
