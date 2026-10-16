import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from reward_wrapper import ShapedReward

# Monitor sits INSIDE the wrapper, so ep_rew_mean shows the true game score
env = ShapedReward(Monitor(gym.make("LunarLander-v3")), time_fee=0.1)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("ppo_lunar_timefee")