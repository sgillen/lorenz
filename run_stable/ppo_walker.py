import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import pybullet_envs
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import time

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

trial_name = input("Trial name: ")


def run_stable(num_steps, save_name):

    env = make_vec_env('Walker2DBulletEnv-v0', n_envs=4)
    model = PPO2(MlpPolicy,
                 env,
                 verbose=2,
                 seed = int(seed),
                 #normalize= True,
                 #policy= 'MlpPolicy',
                 n_steps= 1024,
                 nminibatches= 64,
                 lam= 0.95,
                 gamma= 0.99,
                 noptepochs= 10,
                 ent_coef= 0.0,
                 learning_rate= 2.5e-4,
                 cliprange= 0.1,
                 cliprange_vf= -1,
    )


    model.learn(total_timesteps=num_steps)
    model.save(save_name)

if __name__ == "__main__":
    
    start = time.time()

    proc_list = []
    for seed in np.random.randint(0, 2 ** 32, 8):
        
        #    run_stable(int(8e4), "./data/walker/" + trial_name + "_" + str(seed))
        
        p = Process(
            target=run_stable,
            args=(int(1e7), "./data/walker2/" + trial_name + "_" + str(seed))
        )
        p.start()
        proc_list.append(p)
        
        
    for p in proc_list:
        print("joining")
        p.join()



    print(time.time() - start)
