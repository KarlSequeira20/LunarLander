import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

models = ["ppo_lunar_baseline", "ppo_lunar_timefee", "ppo_lunar_hack", "ppo_lunar_fixed"]

for name in models:
    env = Monitor(gym.make("LunarLander-v3"))
    model = PPO.load(name)
    rewards, lengths = evaluate_policy(
        model, env, n_eval_episodes=50, deterministic=True,
        return_episode_rewards=True,
    )
    mean = sum(rewards) / len(rewards)
    crashes = sum(1 for r, l in zip(rewards, lengths) if r < 0 and l < 1000)
    hovers = sum(l >= 1000 for l in lengths)
    solved = sum(r >= 200 for r in rewards)
    print(f"{name:22s} mean {mean:6.1f} | solved(200+) {solved:2d}/50 | crashes {crashes:2d} | timeouts {hovers:2d}")
    