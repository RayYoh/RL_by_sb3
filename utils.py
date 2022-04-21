import os
import warnings
import argparse
import yaml
import torch
import gym
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import OrderedDict
import numpy as np
from gym import spaces

# For using HER with GoalEnv
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HerReplayBuffer  # noqa: F401
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure, read_csv
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)
ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    # SB3 Contrib,
    # "ars": ARS,
    # "qrdqn": QRDQN,
    # "tqc": TQC,
    # "trpo": TRPO,
}


class Experiment(object):
    def __init__(
        self,
        args: argparse.Namespace,
        env_id: str,
        n_env: int = 1,
        env_kwargs: Optional[Dict[str, Any]] = None,
        n_evaluations: int = 1,
        hyperparams: dict = {},
        save_replay_buffer: bool = False,
        verbose: int = 1,
        vec_env_type: str = "dummy",
        n_eval_envs: int = 1,
    ):
        super(Experiment, self).__init__()
        self.algo = args.algo
        self.env_id = env_id
        self.env_kwargs = {} if env_kwargs is None else env_kwargs
        self.n_timesteps = args.n_timesteps
        self.normalize = False
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = args.seed

        self.vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]

        self.vec_env_kwargs = {}
        # self.vec_env_kwargs = {} if vec_env_type == "dummy" else {"start_method": "fork"}

        # Callbacks
        self.specified_callbacks = []
        self.callbacks = []
        self.save_freq = args.save_freq
        self.eval_freq = args.eval_freq
        self.n_eval_episodes = args.eval_episodes
        self.n_eval_envs = n_eval_envs

        self.n_envs = n_env  # it will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self._hyperparams = hyperparams

        self._is_atari = self.is_atari(env_id)
        self.n_evaluations = n_evaluations
        self.deterministic_eval = not self.is_atari(self.env_id)
        self.device = args.device

        # Logging
        self.log_folder = args.log_folder
        self.verbose = verbose
        self.args = args
        self.save_replay_buffer = save_replay_buffer

        self.save_path = self.log_folder + f"/{self.env_id}_{self.algo}/{self.algo}_{self.seed}/"
        
    def setup_experiment(self) -> Optional[Tuple[BaseAlgorithm, Dict[str, Any]]]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """

        self.create_log_folder()
        self.create_callbacks()

        # Create env to have access to action space for action noise
        n_envs = 1 if self.algo == "ars" else self.n_envs
        env = self.create_envs(n_envs, no_log=False)

         # set up logger
        new_logger = configure(self.save_path, ["stdout", "log", "tensorboard", "csv", "json"])
        # Train an agent from scratch
        model = ALGOS[self.algo](
            env=env,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
            **self._hyperparams,
            )
        model.set_logger(new_logger)

        self._save_config(self._hyperparams)
        return model, self._hyperparams

    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs = {}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        try:
            model.learn(self.n_timesteps, **kwargs)
        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            try:
                model.env.close()
            except EOFError:
                pass
    
    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.save_path

        monitor_kwargs = {}
        # Special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id) or "parking-v0" in self.env_id:
            monitor_kwargs = dict(info_keywords=("is_success",))

        # On most env, SubprocVecEnv does not help and is quite memory hungry
        # therefore we use DummyVecEnv by default
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.seed,
            env_kwargs=self.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=self.vec_env_class,
            vec_env_kwargs=self.vec_env_kwargs,
            monitor_kwargs=monitor_kwargs,
        )

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, gym.spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space
                )

            if wrap_with_vectranspose:
                if self.verbose >= 1:
                    if eval_env:
                        print("Wrapping the eval env in a VecTransposeImage.")
                    else:
                        print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env
    def create_log_folder(self):
        os.makedirs(self.save_path, exist_ok=True)

    def create_callbacks(self):

        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)

            if self.verbose > 0:
                print("Creating test environment")

            eval_callback = EvalCallback(
                self.create_envs(self.n_eval_envs, eval_env=True),
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval,
            )

            self.callbacks.append(eval_callback)
    
    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be use later
        to reproduce an experiment.

        :param saved_hyperparams:
        """
        # Save hyperparams
        with open(os.path.join(self.save_path, "hyperparams.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        # save command line arguments
        with open(os.path.join(self.save_path, "args.yml"), "w") as f:
            ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
            yaml.dump(ordered_args, f)

        print(f"Hyperparams Log path: {self.save_path}")

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics

        :param model:
        """
        print(f"Saving to {self.save_path}")
        model.save(f"{self.save_path}/{self.env_id}")

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            print("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

    
    @staticmethod
    def is_atari(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "AtariEnv" in str(entry_point)

    @staticmethod
    def is_bullet(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "pybullet_envs" in str(entry_point)

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        entry_point = gym.envs.registry.env_specs[env_id].entry_point
        return "gym.envs.robotics" in str(entry_point) or "panda_gym.envs" in str(entry_point)
def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    ) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.

    Valid images: RGB, RGBD, GrayScale with values in [0, 255]

    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return:
    """
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False

def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """
    Check if an image observation space (see ``is_image_space``)
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).

    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).

    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn("Treating image space as channels-last, while second dimension was smallest of the three.")
    return smallest_dimension == 0