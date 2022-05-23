"""Code to evaluate a learned ARS policy.
"""
from enum import Flag
from absl import app
from absl import flags
from absl import logging
from datetime import datetime
import numpy as np
import gym
import os
import pickle
import time
from src.robots import robot
from tqdm import tqdm
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

flags.DEFINE_string('logdir', '/path/to/log/dir', 'path to log dir.')
flags.DEFINE_bool('show_gui', False, 'whether to show pybullet GUI.')
flags.DEFINE_bool('save_video', True, 'whether to save video.')
flags.DEFINE_bool('save_data', True, 'whether to save data.')
flags.DEFINE_integer('num_rollouts', 1, 'number of rollouts.')
flags.DEFINE_integer('rollout_length', 0, 'rollout_length, 0 for default.')
flags.DEFINE_bool('use_real_robot', False, 'whether to use real robot.')
flags.DEFINE_bool('use_gamepad', False,
                  'whether to use gamepad for speed command.')
FLAGS = flags.FLAGS

def main(_):

  # Load config and policy
  config_path = os.path.join(os.path.dirname(FLAGS.logdir),'good', 'config.yaml')
  model_path = os.path.join(FLAGS.logdir,'ppo_policy')
  log_path = FLAGS.logdir
  
  if FLAGS.save_video or FLAGS.save_data:
    log_path = os.path.join(log_path,
                            datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(log_path)

  with open(config_path, 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)

  # Set config
  with config.unlocked():
    config.env_args.show_gui = FLAGS.show_gui
    config.env_args.use_real_robot = FLAGS.use_real_robot
    config.env_args.use_gamepad_speed_command = FLAGS.use_gamepad

  if FLAGS.rollout_length:
    config.rollout_length = FLAGS.rollout_length
  env = config.env_constructor(**config.env_args)

  returns = []
  observations = []
  actions = []
  renders = []
  episode_lengths = []
  max_placement = 0
  tick = 0
  obs = env.reset()
  done = False
  totalr = 0.
  steps = 0
  model = PPO.load(model_path)

  # p = env.pybullet_client
  # if FLAGS.save_video:
  #   video_dir = os.path.join(log_path, 'video.mp4')
  #   log_id = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, video_dir)
 
  states = []
  for t in range(config.rollout_length):
    start_time = time.time()
    action, _ = model.predict(obs)
    observations.append(obs)
    actions.append(action)
    rew = 0

    for _ in range(int(env.config.high_level_dt / env.robot.control_timestep)):
      
      start_time = time.time()
      # obs, step_rew, step_impulse, done, _ = env.step(action, single_step=True)
      obs, step_rew, done, _ = env.step(action, single_step=True)
      
      rew += step_rew
      states.append(
          dict(
              desired_speed=env.get_desired_speed(
                  env.robot.time_since_reset),
              timestamp=env.robot.time_since_reset,
              base_rpy=env.robot.base_orientation_rpy,
              motor_angles=env.robot.motor_angles,
              base_vel=env.robot.base_velocity,
              base_vel_x=env.robot.base_velocity[0],
              base_vels_body_frame=env.state_estimator.com_velocity_body_frame,
              base_rpy_rate=env.robot.base_rpy_rate,
              motor_vels=env.robot.motor_velocities,
              motor_torques=env.robot.motor_torques,
              contacts=env.robot.foot_contacts,
              desired_grf=env.qp_sol,
              reward=step_rew,
              state=obs,
              robot_action=env.robot_action,
              env_action=action,
              gait_generator_phase=env.gait_generator.current_phase.copy(),
              gait_generator_state=env.gait_generator.leg_state,
              gait_normalized_phase=env.gait_generator.normalized_phase[0],
              foot_velocity=env.robot.foot_velocity,
              tick=tick
              # impulse=step_impulse
              ))
      # delta +=1
      tick += 1

      if done:
        break
    totalr += rew
    steps += 1
    if done:
      break

    duration = time.time() - start_time
    if duration < env.robot.control_timestep and not FLAGS.use_real_robot:
      time.sleep(env.robot.control_timestep - duration)

  if FLAGS.save_data:
    pickle.dump(states, open(os.path.join(log_path, 'states_0.pkl'), 'wb'))
    logging.info("Data logged to: {}".format(log_path))
  print(totalr)
  # print("End Phase: {}".format(env.gait_generator.current_phase))
  episode_lengths.append(steps)
  returns.append(totalr)

  print('episode lengths', episode_lengths)
  print('returns', returns)


if __name__ == '__main__':
  app.run(main)
