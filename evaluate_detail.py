import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Usage: python evaluate_detail.py ppo_lunar_hack
model_name = sys.argv[1] if len(sys.argv) > 1 else "ppo_lunar_baseline"

# Plain environment: no custom reward wrapper, so scores are the true game score
env = Monitor(gym.make("LunarLander-v3"))
model = PPO.load(model_name)

rewards, lengths = evaluate_policy(
    model, env, n_eval_episodes=20, deterministic=True,
    return_episode_rewards=True,
)

print(f"Model: {model_name}")
for i, (r, l) in enumerate(zip(rewards, lengths), 1):
    print(f"Episode {i:2d}: reward {r:7.1f}, {l} steps")

mean = sum(rewards) / len(rewards)
print(f"Mean reward: {mean:.1f}")
print("Crashes (reward < 0):", sum(r < 0 for r in rewards))
print("Long episodes (>= 1000 steps, i.e. hovering):", sum(l >= 1000 for l in lengths))