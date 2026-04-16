import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4")
obs, _ = env.reset()

data = []

for _ in range(1000):
    observation = obs["observation"]
    desired_goal = obs["desired_goal"]

    current_pos = observation[:3]
    action = desired_goal - current_pos
    action = action * 5.0
    action = np.concatenate([action, [0]])

    data.append((observation, action))

    obs, _, terminated, truncated, _ = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

np.save("data.npy", data)
env.close()
