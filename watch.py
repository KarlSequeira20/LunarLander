import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3", render_mode="human")
model = PPO.load("ppo_lunar_baseline")

for episode in range(3):
    obs, info = env.reset()
    done = False
    total = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        done = terminated or truncated
    print(f"Episode {episode + 1}: {total:.1f}")

env.close()