import gymnasium as gym

class ShapedReward(gym.Wrapper):
    def __init__(self, env, time_fee=0.1):
        super().__init__(env)
        self.time_fee = time_fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward -= self.time_fee          # my rule: every frame costs a little
        return obs, reward, terminated, truncated, info


class HackableReward(gym.Wrapper):
    """+1 per frame while upright and slow. Intentionally exploitable."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        x, y, vx, vy, angle, spin, leg1, leg2 = obs
        if abs(angle) < 0.25 and abs(vx) < 0.5 and abs(vy) < 0.5:
            reward += 1.0
        return obs, reward, terminated, truncated, info


class FixedReward(gym.Wrapper):
    """Pays for improvement (potential-based), plus a small time fee."""
    def __init__(self, env, time_fee=0.1, scale=5.0, gamma=0.99):
        super().__init__(env)
        self.time_fee, self.scale, self.gamma = time_fee, scale, gamma
        self.prev_phi = 0.0

    @staticmethod
    def phi(obs):
        x, y, vx, vy, angle, spin, leg1, leg2 = obs
        return -(abs(x) + abs(vx) + abs(vy) + abs(angle))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_phi = self.phi(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        new_phi = self.phi(obs)
        reward += self.scale * (self.gamma * new_phi - self.prev_phi) - self.time_fee
        self.prev_phi = new_phi
        return obs, reward, terminated, truncated, info