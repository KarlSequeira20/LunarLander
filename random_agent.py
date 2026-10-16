import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

for episode in range(3):
    obs, info = env.reset()
    total_reward = 0
    done = False
    steps = 0

    while not done:
        action = env.action_space.sample()  # random: 0 nothing, 1 left, 2 main, 3 right
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if steps == 1 or steps == 2:
            print(f"Frame {steps}: reward = {reward:.2f}, obs = {obs.round(2)}")

    print(f"Episode {episode + 1} finished: {steps} steps, total reward = {total_reward:.1f}")

env.close()