from multiprocessing import Process
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.run_utils import run_sg, load_workspace
from seagul.rl.sac import sac, SACModel
from seagul.nn import MLP
from seagul.integration import euler
import seagul.envs
import time

#from seagul.integrationx import euler

proc_list = []
env_name = "InvertedPendulum-v2"

env = gym.make(env_name)

# init policy, value fn
input_size = env.observation_space.shape[0]
output_size = 1
layer_size = 32
num_layers = 2
activation = nn.ReLU

base_dir = "/data/mj_pend/"
trial_name = input("Trial name: ")

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()

start = time.time()
for seed in np.random.randint(0, 2 ** 32, 8):
    
    model = SACModel(
        policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation),
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
        q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
        q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
        act_limit=3
    )


    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "total_steps" : 1e6,
        "alpha" : .2,
        "exploration_steps" : 5000,
        "min_steps_per_update" : 500,
        "reward_stop" : 950,
        "env_steps" : env._max_episode_steps,
        "gamma": 1,
        "sgd_batch_size": 64,
        "replay_batch_size" : 256,
        "iters_per_update": 4,
        #"iters_per_update": float('inf'),
    }


    # run_sg(alg_config, sac, "sac bullet defaults", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

    p = Process(
        target=run_sg,
        args=(alg_config, sac, trial_name + "/" + "seed" + str(seed), "sac bullet defaults", base_dir),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()

print(f"experiment complete, total time: {time.time() - start}, saved in {base_dir+trial_name}")

