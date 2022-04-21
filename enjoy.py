import os
import time
import argparse

import gym
import numpy as np

from stable_baselines3 import PPO, SAC, TD3
import panda_gym

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Enjoy an RL agent trained using Stable Baselines3"
    )
    parser.add_argument(
        "--algo",
        help="RL Algorithm (Soft Actor-Critic by default)",
        default="ppo",
        type=str,
        required=False,
        choices=["ppo","sac", "td3"],
    )
    parser.add_argument(
        "--env", type=str, default="Swimmer-v3", help="environment ID"
    )
    parser.add_argument(
        "-n", "--n-episodes", help="Number of episodes", default=5, type=int
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        default=False,
        help="Do not render the environment",
    )
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=True,
        help="Load best model instead of last model if available",
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    args = parser.parse_args()

    args.env = "PandaReach-v2"
    env_id = args.env
    # Create an env similar to the training env
    env = gym.make(env_id, render=True)

    # Enable GUI
    if not args.no_render:
        env.render()
    args.algo = "sac"
    algo = {
        "ppo": PPO,
        "sac": SAC,
        "td3": TD3,
    }[args.algo]
    args.seed = 4092870112
    # We assume that the saved model is in the same folder
    save_path = f"./data/{env_id}_{args.algo}/{args.algo}_{args.seed}.zip"

    if not os.path.isfile(save_path) or args.load_best:
        print("Loading best model")
        # Try to load best model
        save_path = os.path.join(f"./data/{env_id}_{args.algo}/{args.algo}_{args.seed}", "best_model.zip")

    # Load the saved model
    model = algo.load(save_path, env=env)

    try:
        # Use deterministic actions for evaluation
        episode_rewards, episode_lengths = [], []
        for _ in range(args.n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _info = env.step(action)
                episode_reward += reward

                episode_length += 1
                if not args.no_render:
                    env.render(mode="human")
                    dt = 1.0 / 240.0
                    time.sleep(dt)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(
                f"Episode {len(episode_rewards)} reward={episode_reward}, length={episode_length}"
            )

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        mean_len, std_len = np.mean(episode_lengths), np.std(episode_lengths)

        print("==== Results ====")
        print(f"Episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode_length={mean_len:.2f} +/- {std_len:.2f}")
    except KeyboardInterrupt:
        pass

    # Close process
    env.close()
