from multiprocessing import Process
import torch.nn as nn
import numpy as np
import gym
from seagul.rl.run_utils import run_sg
from seagul.rl.sac import sac, SACModel
from seagul.nn import MLP
from seagul.integration import euler
import seagul.envs

#from seagul.integrationx import euler

proc_list = []
trial_num = input("What trial is this?\n")
#env_name = "Walker2DBulletEnv-v0"
env_name = "Walker2d-v2"

env = gym.make(env_name)

# init policy, value fn
input_size = env.observation_space.shape[0]
output_size = env.action_space.shape[0]
layer_size = 256
num_layers = 2
activation = nn.ReLU


for seed in np.random.randint(0, 2 ** 32, 8):
    
    model = SACModel(
        policy = MLP(input_size, output_size * 2, num_layers, layer_size, activation),
        value_fn = MLP(input_size, 1, num_layers, layer_size, activation),
        q1_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
        q2_fn = MLP(input_size+output_size, 1, num_layers, layer_size, activation),
        act_limit=1
    )

    alg_config = {
        "env_name": env_name,
        "model": model,
        "seed": int(seed),  # int((time.time() % 1)*1e8),
        "total_steps" : 1e6,
        "alpha" : .01,
        "exploration_steps" : 1000,
        "min_steps_per_update" : 1,
        "env_steps" : env._max_episode_steps,
        "gamma": 1,
        "sgd_batch_size": 256,
        "replay_batch_size" : 256,
        "iters_per_update": float('inf'),
    }

    #run_sg(alg_config, sac, run_desc="sac bullet defaults", "debug", "/data/" + trial_num + "/" + "seed" + str(seed))

    p = Process(
        target=run_sg,
        args=(alg_config, sac,  f"/data_walker/mj{trial_num}/{str(seed)}", "same params as invertedPendulum",""),
    )
    p.start()
    proc_list.append(p)

for p in proc_list:
    print("joining")
    p.join()
