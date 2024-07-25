import abc

import time
import torch
from torch_robotics.torch_utils.torch_utils import tensor_linspace_v1

from corallab_lib import Task


class StraightLinePlanner:

    def __init__(
            self,
            task : Task = None,
            n_support_points : int = 64,
            # tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        self.task = task

        self.n_support_points = n_support_points
        # self.tensor_args = tensor_args

    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs
    ):
        """
        Make straight line trajectory between start and goal.
        """
        traj = tensor_linspace_v1(
            start, goal, steps=self.n_support_points
        ).T

        traj = traj.repeat(n_trajectories, 1, 1)

        return traj, {}

    def render(self, ax, **kwargs):
        pass
