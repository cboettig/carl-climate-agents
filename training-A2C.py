import gym
import gym_climate
import gym_conservation
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env = make_vec_env("ays-v0", n_envs=4)
model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=1000)

model.save("ays-v0-A2C-carl")
