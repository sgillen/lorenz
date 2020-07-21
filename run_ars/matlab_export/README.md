# Post Processed RL Trajectories
Sean Gillen, July 13th 2020

Organization: files are organized as /environment/postprocessor/seed/run.mat

Environment is the id for the OpenAI gym environment.

Postprocessor is the function applied to rewards after each rollout, standard == no processing, rmul == encouraging higher "roughness", vardiv == "encouraging lower fractal dimension". 

Seed is the rng seed for the training process.

Run is a single rollout, each starting from a different initial condition.

.mat files contain obs == Observations, act == Actions, rew == Rewards. All units should be treated as arbitrary, although we can probably back them out if we need to.