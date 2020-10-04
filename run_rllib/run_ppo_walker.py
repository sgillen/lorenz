import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
import seagul.envs

config = ppo.DEFAULT_CONFIG.copy()
config["num_workers"] = 1
config["num_envs_per_worker"] = 1
config["lambda"] = 0.2
config["gamma"] = 0.99
config["num_gpus"] = 0
config["eager"] = False
config["model"]["free_log_std"] = True
config["lr"] = 0.0001
config["kl_coeff"] = 1.0
config["num_sgd_iter"] = 30
config["batch_mode"] = "truncate_episodes"
config["observation_filter"] = "MeanStdFilter"
config["sgd_minibatch_size"] = 64
config["train_batch_size"] = 2048
config["vf_clip_param"] = 10
env_name =  "HalfCheetah-v2"
config["env"] = env_name
config["no_done_at_end"] = True
config["model"]["fcnet_hiddens"] = []

ray.init(
    num_cpus=24,
    memory=int(8e9),
    object_store_memory=int(4e9),
    driver_object_store_memory= int(2e9)
)

analysis = tune.run(
    ppo.PPOTrainer,
    config=config,
    stop={"timesteps_total": 2e6},
    num_samples=4,
    local_dir="../data/tune2/cheetah",
    checkpoint_at_end=True,
)
