# %%
import gym
import os
from stable_baselines.results_plotter import load_results
from seagul.plot import smooth_bounded_curve
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines import PPO2 as ALGO
import time
import seagul.envs
import pybullet_envs

#script_path = os.path.realpath(__file__).split("/")[:-1]
#script_path = "/".join(script_path) + "/"

env_name = "linear_z-v0"
env = gym.make(env_name)


def do_rollout_stable(init_point=None):
    env.seed(int(time.time()))

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = []
    obs1_list = []
    rews_list = []
    xyz_list = []
    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    while not done:
        acts = model.predict(obs)[0]

        obs, rew, done, out = env.step(acts)

        # env.render()
        obs1_list.append(obs)
        obs = torch.as_tensor(obs, dtype=dtype)

        acts_list.append(torch.as_tensor(acts))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        cur_step += 1

    ep_obs1 = torch.tensor(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list)

    return ep_obs1, ep_acts, ep_rews


fig, ax = plt.subplots(1,1)

#log_dir = script_path + './walker_log'
# try:
df_list = []
model_list = []
seed_list = []
min_length = float('inf')

trial_path = "/home/sgillen/work/lorenz/run_stable/data/linear_z/ppo/ppo_cmp2"
for entry in os.scandir(trial_path):
    df = load_results(entry.path)
    seed_list.append(entry.path.split("/")[-1])


    if len(df['r']) < min_length:
       min_length = len(df['r'])

    df_list.append(df)
    model_list.append(ALGO.load(entry.path + "/model.zip"))

min_length = int(min_length)
rewards = np.zeros((min_length, len(df_list)))

for i, df in enumerate(df_list):
   rewards[:, i] = np.array(df['r'][:min_length])

smooth_bounded_curve(rewards)
plt.show()
#
#
# ax.grid()
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# #fig.savefig(script_path + '../figs/reward.png')
# plt.show()


# %%
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

from seagul.integration import euler

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

render=False
model = model_list[0]

try:
    env = gym.make(env_name, **env_config)
except error:
    pass

env.num_steps = 1000

ep_obs1, ep_acts, ep_rews = do_rollout_stable()

plt.plot(ep_obs1); plt.show()

# %%

#model = ALGO.load("/home/sgillen/work/third_party/rl-baselines-zoo/trained_agents/td3/Walker2DBulletEnv-v0.pkl")
#model = model_list[2]

from scipy.io import savemat
trial = 0 
#for model, seed in zip(model_list, seed_list):
for model, seed in zip([model_list[4]], [seed_list[4]]):
    for trial in range(10):
        env._max_episode_steps = 100000
        obs_hist, act_hist, rew_hist, xyz_hist, contact_list, total_steps = do_rollout_debug()
        print(f"{seed}: {total_steps}")

        savemat(f"/home/sgillen/work/lorenz/run_stable/data_katie/agent-{seed}/ic-{trial}.mat", {"obs_hist":np.array(obs_hist),"act_hist":np.array(act_hist),"xyz_hist":xyz_hist,"contact_list":contact_list, "total_steps":total_steps})

        print(f"reward sum: {sum(rew_hist)}, reward len: {len(rew_hist)}")
    
        #t = np.array([i*.01 for i in range(obs_hist.shape[0])])



    #plt.plot(t, xyz_hist)

    #plt.axhline(np.pi/2, -1, 11,color='k',  linestyle='dashed', alpha=.5)
    #plt.axhline(0, -1, 11, color='k', linestyle='dashed', alpha=.5)

    #plt.title('States')
    #plt.xlabel('Time (seconds)')
    #plt.ylabel('Angle (rad)')
    #$plt.legend(['th1', 'th2'])
    #plt.grid()
    #plt.savefig(script_path + '../figs/obs_hist.png')
    #plt.show()
 
     # t = np.array([i*2 for i in range(act_hist.shape[0])])
    # plt.step(t, act_hist, 'k')
    # plt.title('Actions')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Torque (Nm)')
    # plt.grid()
    # #plt.savefig(script_path + '../figs/act_hist.png')
    # plt.show(); plt.figure()


# %%
