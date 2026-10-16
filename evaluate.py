import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("LunarLander-v3"))
model = PPO.load("ppo_lunar_baseline")

mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"Mean reward over 20 episodes: {mean:.1f} +/- {std:.1f}")
