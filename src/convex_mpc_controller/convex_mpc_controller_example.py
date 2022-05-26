"""Example of MPC controller on A1 robot."""
from absl import app
from absl import flags
from absl import logging

import time
import numpy as np
import scipy
import pickle

from src.robots import gamepad_reader
from src.convex_mpc_controller import locomotion_controller

flags.DEFINE_string("logdir", None, "where to log trajectories.")
flags.DEFINE_bool("use_real_robot", False,
                  "whether to use real robot or simulation")
flags.DEFINE_bool("use_joystick", False,
                  "whether to use joystick or pre-defined trajectory")
flags.DEFINE_bool("show_gui", False, "whether to show GUI.")
flags.DEFINE_float("max_time_secs", 1., "maximum time to run the robot.")

FLAGS = flags.FLAGS

speed_profile = (np.array([0, 15, 20, 21, 22]),
                 np.array([[0., 0., 0., 0.], [1.8, 0., 0., 0.],
                           [1.8, 0., 0., 0.], [0., 0., 0., 0.],
                           [0., 0., 0., 0.]]))
x, y = speed_profile
get_desired_speed = scipy.interpolate.interp1d(
    x, y, kind="linear", fill_value="extrapolate", axis=0)

def _update_controller(controller, gamepad):
  
  # Update speed
  if FLAGS.use_joystick:
    lin_speed, rot_speed = gamepad.speed_command
  else:
    desired_speed = get_desired_speed(controller.time_since_reset)
    lin_speed, rot_speed = desired_speed[:3], desired_speed[3:]

  controller.set_desired_speed(lin_speed, rot_speed)

  if (gamepad.estop_flagged) and (controller.mode !=
                                  locomotion_controller.ControllerMode.DOWN):
    controller.set_controller_mode(locomotion_controller.ControllerMode.DOWN)

  # Update controller moce
  controller.set_controller_mode(gamepad.mode_command)

  # Update gait
  controller.set_gait(gamepad.gait_command)  

def main(argv):
  del argv  # unused

  gamepad = gamepad_reader.Gamepad(vel_scale_x=1,
                                   vel_scale_y=1,
                                   vel_scale_rot=1,
                                   max_acc=0.3)
  controller = locomotion_controller.LocomotionController(
      FLAGS.use_real_robot, FLAGS.show_gui)
  try:
    start_time = controller.time_since_reset
    current_time = start_time
    while current_time - start_time < FLAGS.max_time_secs:
      current_time = controller.time_since_reset
      time.sleep(0.05)
      # ctime = time.time()
      _update_controller(controller, gamepad)
      # actual_time = time.time() - ctime
      # print("{:.10f}".format(actual_time))
      if not controller.is_safe:
        gamepad.flag_estop()

  finally:
    gamepad.stop()
    controller.set_controller_mode(
        locomotion_controller.ControllerMode.TERMINATE)


if __name__ == "__main__":
  app.run(main)
