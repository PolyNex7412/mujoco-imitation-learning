import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4", render_mode="human")
obs, _ = env.reset()

for _ in range(1000):
    observation = obs["observation"]
    desired_goal = obs["desired_goal"]

    current_pos = observation[:3]
    action = desired_goal - current_pos
    action = action * 5.0
    action = np.concatenate([action, [0]])

    obs, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
