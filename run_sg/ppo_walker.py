from multiprocessing import Process

env_name = "linear_z-v0"
import torch.nn as nn
import numpy as np

# init policy, value fn
input_size = 4
output_size = 2
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


for seed in np.random.randint(0, 2 ** 32, 8):
    policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation)
    value_fn = MLP(input_size, 1, num_layers, layer_size, activation)

    model = PPOModel(
        policy=policy,
        value_fn=value_fn,
        fixed_std=False
    )
    
    len_schedule = np.asarray([50, 1000])
    sched_length = len_schedule.shape[0]
    x_vals = np.linspace(0, 2e6, sched_length)
    len_lookup = lambda steps: np.interp(steps, x_vals, len_schedule)

    alg_config = {
        "env_name": env_name,
        "model": model,
        "act_var_schedule": [var],
        "len_lambda" : len_lookup,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "total_steps": 2e6,
        "epoch_batch_size": 1024,
        "pol_batch_size": 512,
        "val_batch_size": 1024,
        "lam": .2,
        "gamma": .95,
        "normalize_return": False,
        "env_config": env_config,
        "pol_epochs" : 50,
        "val_epochs" : 30,
    }

    # run_sg(alg_config, ppo, "ppo", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

    p = Process(
        target=run_sg,
        args=(alg_config, ppo, "ppo", "", "/data/rew_rad/" + trial_num + "/" + "seed" + str(seed)),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
