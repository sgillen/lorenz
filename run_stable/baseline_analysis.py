# %%
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
    
    jd = env.unwrapped.robot.jdict
    
    

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


def do_rollout_debug(init_point=None):
    env.seed(int(time.time()))
        
    look_green = True
    look_red= True
    
    if init_point is not None:
        obs = env.reset(init_point)
    else:
        obs = env.reset()

    green_thigh_joint = env.unwrapped.robot.jdict["thigh_joint"]
    red_thigh_joint = env.unwrapped.robot.jdict["thigh_left_joint"]
    
    
    obs = torch.as_tensor(obs, dtype=torch.float32)

    acts_list = []
    obs1_list = []
    rews_list = []
    xyz_list = []
    contact_list = []
    step_list = []
    dtype = torch.float32
    act_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    done = False
    cur_step = 0

    total_steps = 0


    while not done:
        acts = model.predict(obs)[0]
        green_foot_contact = env.unwrapped.robot.feet_contact[0]
        red_foot_contact = env.unwrapped.robot.feet_contact[1]
       
        obs, rew, done, out = env.step(acts)
        if (render):
            env.env.camera_adjust()
            print(f"foot_contact: {red_foot_contact}")
            print(f"f0 contact list: {len (env.unwrapped.robot.feet[0].contact_list())}")
            print(f"f1 contact list: {len (env.unwrapped.robot.feet[1].contact_list())}")
            time.sleep(.02)


        if look_green:
            if green_thigh_joint.get_position() > 0 and green_foot_contact:
                total_steps += 1
                #print("step taken, green foot down")
                look_green = False
                look_red = True


        if look_red:
            if red_thigh_joint.get_position() > 0 and red_foot_contact:
                total_steps += 1
                #print("step taken, red foot down")
                look_green = True
                look_red = False


        # env.render()
        obs1_list.append(obs)
        obs = torch.as_tensor(obs, dtype=dtype)
        xyz_list.append(env.env.robot.body_xyz)

        acts_list.append(torch.as_tensor(acts))
        rews_list.append(torch.as_tensor(rew, dtype=dtype))
        contact_list.append(env.unwrapped.robot.feet_contact.copy())
        cur_step += 1


    ep_obs1 = torch.tensor(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list)

    # print(f"total_steps {total_steps}")
    return ep_obs1, ep_acts, ep_rews, xyz_list, contact_list, total_steps


fig, ax = plt.subplots(1,1)

log_dir = script_path + './walker_log'
# try:
df_list = []
model_list = []
seed_list = []
min_length = float('inf')

trial_path = "/home/sgillen/work/lorenz/run_stable/data2/zoo_td3_mon"
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

#smooth_bounded_curve(rewards[:min_length], ax=ax)


ax.grid()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#fig.savefig(script_path + '../figs/reward.png')
plt.show()


# %%

render=False
model = model_list[-1]

try:
    env = gym.make("Walker2DBulletEnv-v0", render=render)
except error:
    pass

env.num_steps = 1000

#do_rollout_debug()

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