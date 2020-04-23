import gym
import os
from stable_baselines.results_plotter import load_results
from seagul.plot import smooth_bounded_curve
import matplotlib.pyplot as plt
import numpy as np
import torch
import pybullet_envs

script_path = os.path.realpath(__file__).split("/")[:-1]
script_path = "/".join(script_path) + "/"

env_name = "Walker2DBulletEnv-v0"
env = gym.make(env_name, render=True)


def do_rollout_stable(init_point=None):
    model.observation_space = env.observation_space
    #td3_model = TD3.load(script_path + "../rl-baselines-zoo/baseline_log2/td3/su_acrobot_cdc-v0_2/su_acrobot_cdc-v0.zip")

    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = []
    obs1_list = []
    rews_list = []

    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

#    import ipdb; ipdb.set_trace()

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

# %%
fig, ax = plt.subplots(1,1)

log_dir = script_path + './walker_log'
# try:
df_list = []
min_length = float('inf')

trial_path = "/home/sgillen/work/lorenz/run_stable/data/walker2"
for entry in os.scandir(trial_path):
    df = load_results(entry.path)

    if len(df['r']) < min_length:
        min_length = len(df['r'])

    df_list.append(df)

min_length = int(min_length)
rewards = np.zeros((min_length, len(df_list)))

for i, df in enumerate(df_list):
    rewards[:, i] = np.array(df['r'][:min_length])

smooth_bounded_curve(rewards[:min_length], ax=ax)
    #
    # except:
    #     print(algo.path, "did not work")

ax.grid()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#fig.savefig(script_path + '../figs/reward.png')
plt.show()

# %%
from stable_baselines import PPO2 as ALGO

#model = ALGO.load("/home/sgillen/work/third_party/rl-baselines-zoo/trained_agents/td3/Walker2DBulletEnv-v0.pkl")
model = ALGO.load("/home/sgillen/work/lorenz/run_stable/data/walker0.zip")

obs_hist, act_hist, rew_hist = do_rollout_stable()

t = np.array([i*2 for i in range(act_hist.shape[0])])
plt.step(t, act_hist, 'k')
plt.title('Actions')
plt.xlabel('Time (seconds)')
plt.ylabel('Torque (Nm)')
plt.grid()
#plt.savefig(script_path + '../figs/act_hist.png')
plt.show(); plt.figure()

t = np.array([i*.01 for i in range(obs_hist.shape[0])])
plt.plot(t, obs_hist[:,0],'k')
plt.plot(t, obs_hist[:,1],'r')

plt.axhline(np.pi/2, -1, 11,color='k',  linestyle='dashed', alpha=.5)
plt.axhline(0, -1, 11, color='k', linestyle='dashed', alpha=.5)

plt.title('States')
plt.xlabel('Time (seconds)')
plt.ylabel('Angle (rad)')
plt.legend(['th1', 'th2'])
plt.grid()
#plt.savefig(script_path + '../figs/obs_hist.png')
plt.show()
