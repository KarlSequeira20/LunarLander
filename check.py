import gymnasium as gym
import stable_baselines3
import Box2D

env = gym.make("LunarLander-v3")  
obs, info = env.reset()
print("Observation:", obs)
print("All good!")
