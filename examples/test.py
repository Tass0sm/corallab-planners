import numpy as np

import corallab_lib
from corallab_lib import Robot, Env, Task

import corallab_planners
from corallab_planners import Planner

corallab_lib.backend_manager.set_backend("torch_robotics")
corallab_planners.backend_manager.set_backend("ompl")

robot = Robot("RobotPointMass")
env = Env("EnvSquare2D")
task = Task("PlanningTask", robot=robot, env=env)
planner = Planner(
    "RRT",
    task = task
)

start = np.array([-0.7, 0.7])
goal = np.array([0.7, 0.7])
path = planner.solve(start, goal)
