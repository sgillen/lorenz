from seagul.rl.ars.ars_pipe import ars
from seagul.nn import MLP
import torch
import matplotlib.pyplot as plt
from seagul.mesh import variation_dim
from seagul.mesh import mesh_dim
import time
import copy
import gym
import xarray as xr
import numpy as np
import pandas as pd
import os
import seagul.envs
from seagul.integration import euler
from sklearn.decomposition import PCA

def identity(rews,obs,acts):
    return rews

def madodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=1)

def variodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=2)

def radodiv(rews, obs, acts):
    return rews/variation_dim(obs, order=.5)

# def vardiv(rews, obs, acts):
#     return rews/variation_dim(obs)

# def varmul(rews, obs, acts):
#     return rews*variation_dim(obs)

def rough_measure(x):
    d = torch.stack([x[t,:] - x[t-1,:] for t in range(1,x.shape[0])])
    return d.std()#$/(torch.abs(d.mean()))

def rmul(rews, obs, acts):
    return rews*rough_measure(obs)

def rdiv(rews, obs, acts):
    return rews/rough_measure(obs)

def pcastd(rews, obs, acts):
    pca = PCA()
    pca.fit((obs - obs.mean())/obs.std())
    return rews*pca.explained_variance_ratio_.std()

def mdim_mul(rews, obs, acts):
    m,_,_,_ = mesh_dim(obs)
    return m*rews

def mdim_div(rews, obs, acts):
    m,_,_,_ = mesh_dim(obs)
    return rews/m

def cdim_mul(rews, obs, acts):
    _,c,_,_ = mesh_dim(obs)
    return c*rews

def cdim_div(rews, obs, acts):
    _,c,_,_ = mesh_dim(obs)
    return rews/c
   



env_names = ["Walker2d-v2"]#,  "Hopper-v2", "HalfCheetah-v2"]#,  "linear_z2d-v0"]
post_fns = [cdim_div]#, mdim_div]
#post_fns = [cdim_div]

torch.set_default_dtype(torch.float64)
num_experiments = len(post_fns)
num_seeds = 5
num_epochs = 750
n_workers = 24; n_delta = 60; n_top = 20; exp_noise=.025

save_dir = "./datacdim0/"
os.makedirs(save_dir)

import time
start = time.time()


env_config = {}


# def reward_fn_2d(s):
#     if s[2] > 0:
#         if s[0] >= 0 and s[1] >= 0:
#             reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
#             #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2) - 5)**2,0,5)
#             s[2] = -10
#         else:
#             reward = 0.0

#     elif s[2] < 0:
#         if s[0] <= 0 and s[1] <= 0:
#             reward = np.clip(np.sqrt(s[0]**2 + s[1]**2),0,10)
#             #reward = 5 - np.clip(np.abs(np.sqrt(s[0]**2 + s[2]**2)**2 - 5),0,5)
#             s[2] = 10
#         else:
#             reward = 0.0

#     return reward, s


for env_name in env_names:
        env = gym.make(env_name, **env_config)
        in_size = env.observation_space.shape[0]
        out_size = env.action_space.shape[0]
        policy_dict =  {fn.__name__:[] for fn in post_fns}
        
        rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                    dims = ("post", "trial", "epoch"),
                    coords = {"post": [fn.__name__ for fn in post_fns]})

        post_rewards = xr.DataArray(np.zeros((num_experiments, num_seeds, num_epochs)),
                    dims = ("post", "trial", "epoch"),
                    coords = {"post": [fn.__name__ for fn in post_fns]})

        data = xr.Dataset(
            {"rews" : rewards,
            "post_rews" : post_rewards},
            coords = {"post": [fn.__name__ for fn in post_fns]},
            attrs  = {"policy_dict":policy_dict, "post_fns":post_fns, "env_name":env_name,
                      "hyperparams":{"num_experiments":num_experiments, "num_seeds":num_seeds, "num_epochs":num_epochs, "n_workers":n_workers, "n_delta":n_delta, "n_top":n_top, "exp_noise":exp_noise},
                      "env_config":env_config})             
            

        for post_fn in post_fns:
            for i in range(num_seeds):
                policy = MLP(in_size,out_size,0,0)
                policy, r_hist, lr_hist = ars(env_name, policy, num_epochs, n_workers=n_workers, n_delta=n_delta, n_top=n_top, exp_noise=exp_noise, postprocess=post_fn, env_config=env_config)
                print(f"{env_name}, {post_fn.__name__}, {i}, {time.time() - start}")
                data.policy_dict[post_fn.__name__].append(copy.deepcopy(policy))
                data.rews.loc[post_fn.__name__,i,:] = lr_hist
                data.post_rews.loc[post_fn.__name__,i,:] = r_hist

        torch.save(data, f"{save_dir}{env_name}.xr")
