"""Train gait policies using CMAES"""
from absl import app
from absl import logging
from absl import flags

from datetime import datetime
from ml_collections.config_flags import config_flags
import numpy as np
import torch as th
import os
import ray
import tensorflow as tf
import time
from src.agents.ppo import logz
# from src.agents.ppo import rollout_server
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback


config_flags.DEFINE_config_file(
    'config', 'locomotion/agents/ppo/configs/gait_change_deluxe.py',
    'experiment configuration.')
flags.DEFINE_string('experiment_name', 'deluxe_cmaes', 'expriment_name')
flags.DEFINE_integer('random_seed', 1000, 'random seed')
FLAGS = flags.FLAGS

# policy architecture
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=[dict(pi=[64, 64], vf=[64, 64])])

def main(_):
  config = FLAGS.config
  logdir = os.path.join(config.logdir, 'PPO',
                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
  if not (os.path.exists(logdir)):
    os.makedirs(logdir)
  config.logdir = logdir

  # num_of_actors = 32
  env = config.env_constructor(**config.env_args)

  logz.configure_output_dir(config.logdir)
  logz.save_params(config)

  env.reset()
  checkpoint_callback = CheckpointCallback(
      save_freq=100, save_path=logdir, name_prefix='ppo_policy')

  model = PPO("MlpPolicy", env, n_steps=100 ,policy_kwargs=policy_kwargs, verbose=1)
  #800 / 100000
  # start_time = time.time()
  # sum_reward = 0
  logger = configure(logdir,["stdout","log","tensorboard"])
  model.set_logger(logger)
  model.learn(total_timesteps=1000, callback=checkpoint_callback)

  # model.learn(total_timesteps=200000, callback=checkpoint_callback)

  model_dir = os.path.join(logdir,'ppo_policy_final')
  model.save(model_dir)

if __name__ == "__main__":
  app.run(main)
