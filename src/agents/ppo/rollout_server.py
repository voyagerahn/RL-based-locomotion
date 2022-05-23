"""Evaluates policy for CMAES."""
"""Train gait policies using CMAES"""
from absl import logging

from datetime import datetime
import numpy as np
import os
import ray

from src.agents.ppo import logz


@ray.remote
class RolloutServer(object):
  def __init__(self, server_id, config):
    self.server_id = server_id
    self.config = config
    self.env = config.env_constructor(**config.env_args)
    policy_params = {
        'ob_filter': config.filter,
        'ob_dim': self.env.observation_space.low.shape[0],
        'ac_dim': self.env.action_space.low.shape[0]
    }
    self.policy = config.policy_constructor(self.config, policy_params,
                                            self.env.observation_space,
                                            self.env.action_space)

  
  def eval_policy(self, step, policy_weight, eval=False):
    self.policy.update_weights(policy_weight)
    if eval:
      np.random.seed(0)
    else:
      np.random.seed(step * 100 + self.server_id)
    state = self.env.reset()
    sum_reward = 0
    for i in range(self.config.rollout_length):
      action, neglogp, _ = agent.step(state)
      observation, rew, done, _ = self.env.step(action)

      agent.memory.store_transition(state, action, (rew + 8) / 8, neglogp)

      state = observation

      sum_reward += rew
      if (i + 1) % 32 == 0 or i == config.rollout_length - 1:
          _, _, last_value = agent.step(observation)
          agent.learn(last_value, done)
      if done:
        break
    return sum_reward

  def get_weights_plus_stats(self):
    return self.policy.get_weights_plus_stats()


    for i in range(config.rollout_length):
      action, neglogp, _ = agent.step(state)
      observation, rew, done, _ = env.step(action)

      agent.memory.store_transition(state, action, (rew + 8) / 8, neglogp)

      state = observation
      sum_reward += rew
      
      if (i + 1) % 32 == 0 or i == config.rollout_length - 1:
          _, _, last_value = agent.step(observation)
          agent.learn(last_value, done)
      
      if done:
        break