# %%
from stable_baselines.results_plotter import load_results
from seagul.plot import smooth_bounded_curve
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
from gym.wrappers import TimeLimit
import pybullet_envs
import time
import torch
from stable_baselines import SAC, TD3, PPO2, A2C, TRPO
import sys

sys.path.append('/home/sgillen/work/third_party/rl-baselines-zoo/')
#from stable_baselines import utils

script_path = os.path.realpath(__file__).split("/")[:-1]
script_path = "/".join(script_path) + "/"


class TimeFeatureWrapper(gym.Wrapper):
    """
    Add remaining time to observation space for fixed length episodes.
    See https://arxiv.org/abs/1712.00378 and https://github.com/aravindr93/mjrl/issues/13.
    :param env: (gym.Env)
    :param max_steps: (int) Max number of steps of an episode
        if it is not wrapped in a TimeLimit object.
    :param test_mode: (bool) In test mode, the time feature is constant,
        equal to zero. This allow to check that the agent did not overfit this feature,
        learning a deterministic pre-defined sequence of actions.
    """
    def __init__(self, env, max_steps=1000, test_mode=False):
        assert isinstance(env.observation_space, gym.spaces.Box)
        # Add a time feature to the observation
        low, high = env.observation_space.low, env.observation_space.high
        low, high= np.concatenate((low, [0])), np.concatenate((high, [1.]))
        env.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        super(TimeFeatureWrapper, self).__init__(env)

        if isinstance(env, TimeLimit):
            self._max_steps = env._max_episode_steps
        else:
            self._max_steps = max_steps
        self._current_step = 0
        self._test_mode = test_mode

    def reset(self):
        self._current_step = 0
        return self._get_obs(self.env.reset())

    def step(self, action):
        self._current_step += 1
        obs, reward, done, info = self.env.step(action)
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        """
        Concatenate the time feature to the current observation.
        :param obs: (np.ndarray)
        :return: (np.ndarray)
        """
        # Remaining time is more general
        time_feature = 1 - (self._current_step / self._max_steps)
        if self._test_mode:
            time_feature = 1.0
        # Optionnaly: concatenate [time_feature, time_feature ** 2]
        return np.concatenate((obs, [time_feature]))


def do_rollout_debug(init_point=None):
    env.seed(int(time.time())*10)
        
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

#    import ipdb; ipdb.set_trace()

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
        contact_list.append(env.unwrapped.robot.feet_contact)
        cur_step += 1


    ep_obs1 = torch.tensor(obs1_list)
    ep_acts = torch.stack(acts_list)
    ep_rews = torch.stack(rews_list)

    #print(f"total_steps {total_steps}")
    return ep_obs1, ep_acts, ep_rews, total_steps


color_iter = iter(['b', 'g', 'y', 'm', 'c'])
log_dir = script_path + './data_r/walker_r_log2'
legend_list = []

fig, ax = plt.subplots(1,1)
for algo in os.scandir(log_dir):

        df_list = []
        min_length = float('inf')

        for entry in os.scandir(algo.path):
            df = load_results(entry.path)

            if len(df['r']) < min_length:
                min_length = len(df['r'])

            df_list.append(df)

        min_length = int(min_length)
        rewards = np.zeros((min_length, len(df_list)))

        for i, df in enumerate(df_list):
            rewards[:, i] = np.array(df['r'][:min_length])

        print(print(algo.path), rewards[-1, :].mean(), rewards[-1, :].std())
        smooth_bounded_curve(rewards[:min_length], time_steps=[51*i for i in range(min_length)], ax=ax, color=color_iter.__next__())
        print(algo.path)
        legend_list.append(algo.path.split('/')[-1])

plt.legend(legend_list)
ax.grid()
ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
plt.show()



#model = SAC.load("/home/sgillen/work/lorenz/run_stable/walker_r_log2/sac/Walker2DBulletEnv-v0_1/best_model.zip")
render = False
#env = TimeFeatureWrapper(gym.make("Walker2DBulletEnv-v0", render=render))
env = gym.make("Walker2DBulletEnv-v0", render=render)


env.num_steps=int(1e8)
env._max_episode_steps = int(1e8)
env._max_steps = int(1e8)


algo_step_list = []
algo_rew_list = []
for trial in os.scandir("/home/sgillen/work/lorenz/run_stable/data2/zoo_td3_mon"):
    model = TD3.load(trial.path + "/model.zip")
    step_list = []
    rew_list = []
    for _ in range(10):
        obs,acts,rews,steps = do_rollout_debug()
        step_list.append(steps)
        rew_list.append(sum(rews))
        algo_step_list.append(steps)
        algo_rew_list.append(sum(rews))
        print(len(rews))
    print("steps: ",  np.array(step_list).mean(), "+-", np.array(step_list).std() )
    print("rews: ",  np.array(rew_list).mean(), "+-", np.array(rew_list).std() )

print("total steps", np.array(algo_step_list).mean(), "+-", np.array(algo_step_list).std())
print("total rews: ",  np.array(algo_rew_list).mean(), "+-", np.array(algo_rew_list).std())

# %%