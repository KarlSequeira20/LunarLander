import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_baseline", env=env)
model.learn(total_timesteps=500_000, reset_num_timesteps=False)
model.save("ppo_lunar_baseline_1p5M")