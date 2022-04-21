import gym
import atari_py
env = gym.make("Pong-v0")
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)
print(info)

# from gym import envs
# for env in envs.registry.all():
#     print(env.id)
