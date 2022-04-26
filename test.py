# import gym
# import atari_py
# env = gym.make("Pong-v0")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42, return_info=True)
# print(info)

# from gym import envs
# for env in envs.registry.all():
#     print(env.id)

# from stable_baselines3.common.logger import read_csv, read_json
# data_csv = read_csv("./data/CartPole-v0_ppo/ppo_3567743922/progress.csv")
# data_json = read_json("./data/CartPole-v0_ppo/ppo_3567743922/progress.json")
# print(data_csv)
# print(data_json)
import os
log_path = "rl-trained-agents/ppo"
print(os.listdir(log_path))