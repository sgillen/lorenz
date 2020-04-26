import gym
import os
from stable_baselines.results_plotter import load_results
from seagul.plot import smooth_bounded_curve
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines import TD3 as ALGO
import time
import pybullet_envs

script_path = os.path.realpath(__file__).split("/")[:-1]
script_path = "/".join(script_path) + "/"

env_name = "Walker2DBulletEnv-v0"
render = False
env = gym.make(env_name, render=render)


def do_rollout_stable(init_point=None):
    env.seed(int(time.time()))
    #model.observation_space = env.observation_space
    #td3_model = TD3.load(script_path + "../rl-baselines-zoo/baseline_log2/td3/su_acrobot_cdc-v0_2/su_acrobot_cdc-v0.zip")
    #env._max_episode_steps = 5000

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

#    import ipdb; ipdb.set_trace()

    while not done:
        acts = model.predict(obs)[0]

        obs, rew, done, out = env.step(acts)
        if (render):
            env.env.camera_adjust()
            time.sleep(.02)

        # env.render()
        obs1_list.append(obs)
        obs = torch.as_tensor(obs, dtype=dtype)
        xyz_list.append(env.env.robot.body_xyz)

        acts_list.append(torch.as_tensor(acts))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        cur_step += 1

    ep_obs1 = torch.tensor(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list)

    return ep_obs1, ep_acts, ep_rews, xyz_list

# %%
fig, ax = plt.subplots(1,1)

log_dir = script_path + './walker_log'
# try:
df_list = []
model_list = []
min_length = float('inf')

trial_path = "/home/sgillen/work/lorenz/run_stable/data2/zoo_td3_mon"
for entry in os.scandir(trial_path):
    df = load_results(entry.path)


    if len(df['r']) < min_length:
       min_length = len(df['r'])

    df_list.append(df)
    model_list.append(ALGO.load(entry.path + "/model.zip"))

min_length = int(min_length)
rewards = np.zeros((min_length, len(df_list)))

for i, df in enumerate(df_list):
   rewards[:, i] = np.array(df['r'][:min_length])

smooth_bounded_curve(rewards[:min_length], ax=ax)


ax.grid()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#fig.savefig(script_path + '../figs/reward.png')
plt.show()

# %%

#model = ALGO.load("/home/sgillen/work/third_party/rl-baselines-zoo/trained_agents/td3/Walker2DBulletEnv-v0.pkl")
#model = model_list[2]
for model in model_list:
    #df = df_list[2]

    #plt.plot(df['r']); plt.show()

    env._max_episode_steps = 100000
    obs_hist, act_hist, rew_hist, xyz_hist = do_rollout_stable()

    print(f"reward sum: {sum(rew_hist)}, reward len: {len(rew_hist)}")

    # t = np.array([i*2 for i in range(act_hist.shape[0])])
    # plt.step(t, act_hist, 'k')
    # plt.title('Actions')
    # plt.xlabel('Time (seconds)')
    # plt.ylabel('Torque (Nm)')
    # plt.grid()
    # #plt.savefig(script_path + '../figs/act_hist.png')
    # plt.show(); plt.figure()

    t = np.array([i*.01 for i in range(obs_hist.shape[0])])
    plt.plot(t, xyz_hist)

    plt.axhline(np.pi/2, -1, 11,color='k',  linestyle='dashed', alpha=.5)
    plt.axhline(0, -1, 11, color='k', linestyle='dashed', alpha=.5)

    plt.title('States')
    plt.xlabel('Time (seconds)')
    #plt.ylabel('Angle (rad)')
    #$plt.legend(['th1', 'th2'])
    plt.grid()
    #plt.savefig(script_path + '../figs/obs_hist.png')
    plt.show()
