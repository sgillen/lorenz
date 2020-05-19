import gym
import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
import pybullet_envs
from multiprocessing import Process
from stable_baselines.common import make_vec_env
import os
from stable_baselines.bench import Monitor
import time
import seagul.envs.bullet


# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

num_steps = int(2e6)
base_dir = "./data/pbmj/0"
trial_name = input("Trial name: ")

trial_dir = base_dir + trial_name + "/"
base_ok = input("run will be saved in " + trial_dir + " ok? y/n")

if base_ok == "n":
    exit()


def run_stable(num_steps, save_dir):
    physics_params = {
        'fixedTimeStep': 0.002,
        'numSubSteps': 1,
        'numSolverIterations': 50,
        'useSplitImpulse': 1,
        'splitImpulsePenetrationThreshold': -0.03999999910593033,
        'contactBreakingThreshold': 0.02,
        'collisionFilterMode': 1,
        'enableFileCaching': 1,
        'restitutionVelocityThreshold': 0.20000000298023224,
        'erp': 0.0,
        'frictionERP': 0.0,
        'contactERP': 0.0,
        'globalCFM': 0.0,
        'enableConeFriction': 0,
        'deterministicOverlappingPairs': 1,
        'allowedCcdPenetration': 0.04,
        'jointFeedbackMode': 0,
        'solverResidualThreshold': 1e-07,
        'contactSlop': 1e-05,
        'enableSAT': 0,
        'constraintSolverType': 0,
        'reportSolverAnalytics': 1,
    }

    dynamics_params = {
        'lateralFriction': 0.9,
        'restitution': 0.0,
        'rollingFriction': 0.0,
        'spinningFriction': 0.0,
        'contactDamping': -1.0,
        'contactStiffness': -1.0,
        'collisionMargin': 0.0,
        'angularDamping': 0.0,
        'linearDamping': 0.0,
        'jointDamping': .1,
    }

    env_config = {"physics_params":physics_params,
                  "dynamics_params":dynamics_params}

    env = make_vec_env(seagul.envs.bullet.PBMJWalker2dEnv, n_envs=4, monitor_dir="./tmp/", env_kwargs=env_config)
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
    model.save(save_dir + "/model.zip")

if __name__ == "__main__":
    
    start = time.time()

    proc_list = []
    for seed in np.random.randint(0, 2 ** 32, 8):
        
        #    run_stable(int(8e4), "./data/walker/" + trial_name + "_" + str(seed))

        save_dir = trial_dir + "/" + str(seed)
        os.makedirs(save_dir, exist_ok=False)
        p = Process(
            target=run_stable,
            args=(num_steps,save_dir )
        )
        p.start()
        proc_list.append(p)
        
        
    for p in proc_list:
        print("joining")
        p.join()


    print(f"experiment complete, total time: {time.time() - start}, saved in {save_dir}")

