import gym
import numpy as np
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import pybullet_envs
import time
import os
from stable_baselines.common import make_vec_env
from multiprocessing import Process
import seagul.envs.bullet
import json
import pybulletgym
import pybulletgym.envs.mujoco.envs.locomotion.walker2d_env

num_steps = int(1e6)


base_dir = "./data_pbg/td3_default/"
trial_name = input("Trial name: ")

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


env_config={}

def run_stable(num_steps, save_dir):

    env = gym.wrappers.TimeLimit(pybulletgym.envs.mujoco.envs.locomotion.walker2d_env.Walker2DMuJoCoEnv,1000)
    env = make_vec_env(env, n_envs=1, monitor_dir=save_dir, env_kwargs=env_config)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    
    model = TD3(MlpPolicy,
                env,
                action_noise=action_noise,
                verbose=1,
                gamma = 0.99,
                buffer_size= 1000000,
                learning_starts= 10000,
                batch_size= 100,
                learning_rate= 1e-3,
                train_freq= 1000,
                gradient_steps= 1000,
                policy_kwargs={"layers":[400, 300]},
                n_cpu_tf_sess=1,
    )

    model.learn(total_timesteps=num_steps)
    model.save(save_dir + "/model.zip")


if __name__ == "__main__":
    
    start = time.time()

    proc_list = []

    os.makedirs(trial_dir, exist_ok=False)
    with open(trial_dir + "config.json", "w") as config_file:
        json.dump(env_config, config_file)

    
    for seed in np.random.randint(0, 2 ** 32, 8):
        #    run_stable(int(8e4), "./data/walker/" + trial_name + "_" + str(seed))

        save_dir = trial_dir + "/" + str(seed)
        os.makedirs(save_dir, exist_ok=False)

        
        p = Process(
            target=run_stable,
            args=(num_steps, save_dir)
        )

        p.start()
        proc_list.append(p)

    for p in proc_list:
        print("joining")
        p.join()

    print(f"experiment complete, total time: {time.time() - start}, saved in {save_dir}")

