import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_wrapper import HackableReward

# Monitor is inside the wrapper, so ep_rew_mean shows the TRUE game score
env = HackableReward(Monitor(gym.make("LunarLander-v3")))

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lunar_hack")