import sys
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

# Usage: python record.py <model_name or random> <prefix> [episodes]
model_name = sys.argv[1]
prefix = sys.argv[2]
episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 3

env = gym.make("LunarLander-v3", render_mode="rgb_array")
env = RecordVideo(env, video_folder="videos", name_prefix=prefix,
                  episode_trigger=lambda ep: True)
model = None if model_name == "random" else PPO.load(model_name)

for episode in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()
print("Saved to the videos folder.")