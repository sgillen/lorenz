import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
import pybullet_envs

env = gym.make('Walker2d-v2')
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])
n_steps = int(2e6)

model = SAC(MlpPolicy,
            env,
            verbose=1,
            gamma = 0.99,
            buffer_size= 1000000,
            learning_starts= 10000,
            batch_size= 100,
            learning_rate= 1e-3,
            train_freq= 1000,
            gradient_steps= 1000,
            policy_kwargs={"layers":[400, 300], "n_env":1, "n_steps":n_steps, "n_batch":1}
)

model.learn(total_timesteps=n_steps)
model.save("walker_td3")
