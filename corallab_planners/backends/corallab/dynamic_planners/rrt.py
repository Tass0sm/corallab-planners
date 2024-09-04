import math
import sys
from random import random
from copy import copy
from operator import itemgetter

import numpy as np
import torch
import time
import matplotlib.pyplot as plt

from .rrt_base import RRTBase
from corallab_planners.backends.corallab.planners.utils import purge_duplicates_from_traj, extend_path
from corallab_planners.backends.corallab.dynamic_planners.utils import check_motion

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.torch_utils.torch_timer import TimerCUDA


class DTreeNode:

    def __init__(self, config, time=0, parent=None):
        self.config = config
        self.parent = parent
        self.time = time

    def retrace(self):
        sequence = []
        node = self
        while node is not None:
            sequence.append(node)
            node = node.parent
        return sequence[::-1]

    def render(self, ax):
        assert ax is not None, "Axis cannot be None"
        if self.parent is not None:
            if ax.name == '3d':
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot3D([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color='k', linewidth=0.5)
            else:
                x, y = self.config.cpu().numpy(), self.parent.config.cpu().numpy()
                ax.plot([x[0], y[0]], [x[1], y[1]], color='k', linewidth=0.5)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


def random_swap(nodes1, nodes2):
    p = float(len(nodes1)) / (len(nodes1) + len(nodes2))
    swap = (torch.rand(1) < p)
    return swap


def configs(nodes):
    if nodes is None:
        return None
    return list(map(lambda n: n.config, nodes))


class RRT(RRTBase):

    def __init__(
            self,
            problem=None,
            n_iters: int = 30000,
            step_size: float = 0.1,
            n_radius: float = 1.,
            max_time: float = 10.,
            goal_probability: float = 0.2,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            n_pre_samples=10000,
            pre_samples=None,
            **kwargs
    ):
        super(RRT, self).__init__(
            problem = problem,
            n_iters = n_iters,
            step_size = step_size,
            n_radius = n_radius,
            max_time = max_time,
            tensor_args = tensor_args,
            n_pre_samples = n_pre_samples,
            pre_samples = pre_samples
        )

        self.goal_probability = goal_probability

        self.nodes_tree = None
        self.nodes_tree_torch = None

    def solve(
            self,
            start,
            goal,
            **kwargs
    ):
        print_freq = kwargs.get('print_freq', 150)
        debug = kwargs.get('debug', False)

        if (self.problem.static_check_collision(start.reshape(1, 1, -1)) or
            self.problem.static_check_collision(goal.reshape(1, 1, -1))):
            return None

        iteration = -1
        success = False

        self.nodes_tree = [DTreeNode(start, time=0)]
        self.nodes_tree_torch = self.nodes_tree[0].config

        path = None

        breakpoint()

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                if iteration % print_freq == 0 or iteration % (self.n_iters - 1) == 0:
                    if debug:
                        self.print_info(iteration, t.elapsed, success)

                should_target_goal = random() < self.goal_probability or iteration == 0
                target = goal if should_target_goal else self.sample_fn(**kwargs)

                ###############################################################
                # nearest node in Tree to the target node
                nearest = self.get_nearest_node(self.nodes_tree, self.nodes_tree_torch, target)

                # create a safe path from the target node to the nearest node
                # no_max_dist=True
                extended = self.problem.local_motion(nearest.config, target, step=self.step_size).unsqueeze(0)
                time = torch.linspace(nearest.time, nearest.time + 1, extended.shape[1], **self.tensor_args)

                last_valid, last_valid_time = check_motion(extended, time, self.collision_fn)

                # add last node in path to Tree1
                if last_valid is None:
                    continue

                new = DTreeNode(last_valid, time=last_valid_time, parent=nearest)
                self.nodes_tree.append(new)
                self.nodes_tree_torch = torch.vstack((self.nodes_tree_torch, new.config))

                # if the last node in path is the same as the proposed node, the two trees are connected and terminate
                if torch.allclose(new.config, goal):
                    success = True
                    path = new.retrace()
                    break

        if path is not None:
            if len(path) == 1:
                return None, {}

            path_states = torch.stack([p.config for p in path])
            path_time = torch.tensor([p.time for p in path])

            self.print_info(iteration, t.elapsed, success)
            return path_states, {"time": path_time}
        else:
            return None, {}

    def print_info(self, iteration, elapsed_time, success):
        print(f'Iteration: {iteration:5}/{self.n_iters:5} '
              f'| Time: {elapsed_time:.3f} s'
              f'| Nodes: {len(self.nodes_tree)} '
              f'| Success: {success}'
              )

    def render(self, ax, **kwargs):
        for node in self.nodes_tree:
            node.render(ax)
