import argparse
import gym
import yaml
import difflib
import importlib
import numpy as np
import torch.nn as nn

from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.utils import set_random_seed
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train an RL agent using Stable Baselines3")
    parser.add_argument(
        "--algo", help="RL Algorithm (PPO by default)", default="sac", type=str, required=False, choices=list(ALGOS.keys()),
    )
    parser.add_argument("--env", type=str, default="CartPole-v1", help="environment ID")
    parser.add_argument("--n-envs", help="Number of environments for training", default=1, type=int)
    parser.add_argument(
        "-n", "--n-timesteps", help="Overwrite the number of training timesteps", default=int(10000), type=int,
    )
    parser.add_argument(
        "--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=int(1e4), type=int,
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=-1)
    # parser.add_argument('--config', help="configuration file", type=str, default="config.yaml")
    parser.add_argument('--log-folder', help="log folder", type=str, default="./data")
    parser.add_argument(
        "--eval-freq",
        help="Evaluate the agent every n steps (if negative, no evaluation). "
        "During hyperparameter optimization n-evaluations is used instead",
        default=10000,
        type=int,
    )
    parser.add_argument("--eval-episodes", help="Number of episodes to use for evaluation", default=5, type=int)
    parser.add_argument("--n-eval-envs", help="Number of environments for evaluation", default=1, type=int)
    parser.add_argument(
        "--save-replay-buffer", help="Save the replay buffer too (when applicable)", action="store_true", default=False
    )
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="dummy", choices=["dummy", "subproc"])
    parser.add_argument("--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=["panda_gym"],
        help="Additional external Gym environment package modules to import (e.g. gym_minigrid)",
    )
    args = parser.parse_args()
    '''
    if args.config:
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = argparse.Namespace(**opt)
    else:
        raise ValueError(f"No config found.")
    '''
    
    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)
    
    # args.env = "PandaReach-v2"
    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())  # pytype: disable=module-attr
    
    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
            print(closest_match)
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError(f"{env_id} not found in gym registry, you maybe meant {closest_match}?")
    
    
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2 ** 32 - 1, dtype="int64").item()

    set_random_seed(args.seed)

    print("=" * 10, env_id, "=" * 10)
    print(f"Seed: {args.seed}")

    """PandaReach-v2"""
    hyperparams = {
        "ppo": dict(
            policy = "MultiInputPolicy",
            learning_rate=5.49717e-05, 
            n_steps=512, 
            batch_size=32, 
            n_epochs=10, 
            gamma=0.999,
            gae_lambda=0.95, 
            clip_range=0.3,
            ent_coef=0.0554757, 
            vf_coef=0.38782,
            max_grad_norm=0.6,
            policy_kwargs=dict(log_std_init=-2,
                               ortho_init=False,
                               activation_fn=nn.ReLU,
                               net_arch=[dict(pi=[256, 256], 
                               vf=[256, 256])])
        ),
        "sac": dict(
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
            # we have to manually specify the max number of steps per episode
            # max_episode_length=100,
            online_sampling=True,
            ),
            buffer_size=int(1e6),
            learning_rate=1e-3,
            learning_starts=1000,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
                ),
            }[args.algo]
    # hyperparams = {"dqn": dict(policy = "MlpPolicy", buffer_size = 1000,)}[args.algo]
    exp = Experiment(args=args, env_id=env_id, hyperparams=hyperparams)

    results = exp.setup_experiment()
    if results is not None:
        model, saved_hyperparams = results
        if model is not None:
            exp.learn(model)