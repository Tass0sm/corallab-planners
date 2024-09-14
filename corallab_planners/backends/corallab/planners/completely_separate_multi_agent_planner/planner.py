import sys
import argparse
import yaml
import math
import torch
import heapq
import numpy as np
from copy import copy

from math import fabs
from itertools import combinations
from copy import deepcopy

from corallab_lib import DynamicPlanningProblem, Robot
from corallab_planners import Planner

from typing import Callable

from corallab_lib import MotionPlanningProblem

from corallab_lib.backends.torch_robotics.env_impl import TorchRoboticsEnv
from corallab_lib.backends.torch_robotics.motion_planning_problem_impl import TorchRoboticsMotionPlanningProblem
from corallab_lib.backends.pybullet.env_impl import PybulletEnv
from corallab_lib.backends.curobo.env_impl import CuroboEnv

import scipy.interpolate

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.robots import MultiRobot


class CompletelySeparateMultiAgentPlanner:

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            # n_iters: int = 30000,
            # max_time: float = 60.,
            # merge_threshold : int = 10,
            # interpolate_solution: bool = True,
            # interpolate_num: int = 64,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        assert problem.robot.is_multi_agent()

        # self.n_iters = n_iters
        # self.max_time = max_time
        # self.merge_threshold = merge_threshold

        # self.interpolate_solution = interpolate_solution
        # self.interpolate_num = interpolate_num

        self.tensor_args = tensor_args

        self.problem = problem
        self.subrobots = self.problem.robot.get_subrobots()
        self.n_subrobots = len(self.subrobots)
        self.subrobot_dict = {}

        self.subrobot_problems = []
        self.subrobot_planners = []

        # Subrobot planners
        for i, r in enumerate(self.subrobots):

            # TODO: Fix
            if isinstance(problem.env, TorchRoboticsEnv):
                backend = "torch_robotics"
                env_impl = problem.env
            elif isinstance(problem.env, PybulletEnv):
                backend = "pybullet"
                env_impl = PybulletEnv(
                    problem.env.id,
                    connection_mode="DIRECT",
                    hostname=f"DynamicEnv{i}"
                )
            else:
                backend = "curobo"
                env_impl = problem.env

            single_robot_problem = MotionPlanningProblem(
                env_impl=env_impl,
                robot=r,
                backend=backend
            )

            self.subrobot_problems.append(single_robot_problem)



            single_robot_planner = Planner(
                planner_name="RRTConnect",
                problem=single_robot_problem,
                allowed_time=30.0,
                simplify_solution=True,
                interpolate_solution=True,
                backend="ompl"
            )

            self.subrobot_dict[i] = (single_robot_problem, single_robot_planner)

    def _separate_joint_state(self, joint_state):
        states = []

        for i, r in enumerate(self.subrobots):
            subrobot_state = r.get_position(joint_state)
            states.append(subrobot_state)

            joint_state = joint_state[r.get_n_dof():]

        return states

    def _solve_individual_problems(self, start, goal):
        separate_starts = self._separate_joint_state(start)
        separate_goals = self._separate_joint_state(goal)

        sols = {}

        for i, ((problem, planner), start_i, goal_i) in enumerate(zip(self.subrobot_dict.values(), separate_starts, separate_goals)):
            sol, info = planner.solve(start_i, goal_i)

            if sol is None:
                sols[i] = None
                # sol_times[i] = None
            else:
                # time = info["time"]

                if isinstance(sol, list):
                    sol = sol[0]

                # if isinstance(time, list):
                #     time = time[0]

                sols[i] = sol
                # sol_times[i] = time

        return sols # , sol_times


    def _create_joint_solution(self, times, solutions, n_steps, concatenate=False):
        times_l = list(times.values())
        solutions_l = list(solutions.values())

        # combined_times = np.unique(np.concatenate(times_l))
        # combined_times.sort()

        # if self.interpolate_solution:
        #     ts = self._interpolate_times(combined_times)

        #     if ts is not None:
        #         combined_times = ts

        final_time = max([ts[-1] for ts in times_l])
        combined_times = np.linspace(0, final_time, num=n_steps)

        time_solution_iter = map(lambda p: (p[0], p[1]), zip(times_l, solutions_l))
        interpolators = [scipy.interpolate.interp1d(t, s, axis=0, bounds_error=False, fill_value=(s[0], s[-1])) for t, s in time_solution_iter]
        interpolated_solutions = [f(combined_times) for f in interpolators]

        if concatenate:
            joint_solution = np.concatenate(interpolated_solutions, axis=-1)
        else:
            joint_solution = interpolated_solutions

        return combined_times, joint_solution

    @property
    def name(self):
        return f"completely_separate_multi_agent_planner"

    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs
    ):
        start = torch.tensor(start, **self.tensor_args)
        goal = torch.tensor(goal, **self.tensor_args)

        sol_l = []
        info_l = []

        # solve in sequence
        for _ in range(n_trajectories):
            self.reset()

            sol_i, info_i = self._get_single_solution(start, goal)

            sol_l.append(sol_i)
            info_l.append(info_i)

        sol_l = list(filter(lambda x: x is not None, sol_l))
        # info_l = list(filter(lambda x: "joint_times" in x, info_l))

        if len(sol_l) == 0 or len(info_l) == 0:
            return None, {}

        # joint_times = np.stack([info["joint_times"] for info in info_l])
        # joint_solutions = np.stack(sol_l)
        joint_solutions = sol_l

        return joint_solutions, { }

    def _get_single_solution(
            self,
            start,
            goal
    ):
        iteration = -1
        solution = None

        # create root node of constraint tree with an initial path for every individual
        solutions = self._solve_individual_problems(start, goal)
        solutions_l = [solutions[i] for i in range(len(self.subrobots))]

        if any([sol is None for sol in solutions_l]):
            return None, {}
        else:
            joint_solution = np.concatenate(solutions_l, axis=-1)
            return joint_solution, {}

    def reset(self):
        pass
