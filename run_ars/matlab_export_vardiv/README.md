# Post Processed RL Trajectories
Sean Gillen, July 13th 2020

Organization: files are organized as /experiment/postprocessor/seed/run.mat

Experiment is the name of the robot being run, we have mjw == Mujoco walker, mjhp == Mujoco hopper, mjhc == Mujoco half cheetah.

Postprocessor is the function applied to rewards after each rollout, standard == no processing, rmul == encouraging higher "roughness". hc5 also includes varmul vardiv and rdiv. varmul and vardiv are encouraging higher or lower fractional dimension as determined by a madogram estimator.

Seed is the rng seed for the entire training process.

Run is a single rollout, each starting from a different initial condition.

.mat files contain obs == Observations, act == Actions, rew == Rewards. All units should be treated as arbitrary, although we can probably back them out if we need to.