import os
import abc
import time
import pickle
from pathlib import Path

import numpy as np
import torch
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.torch_utils.torch_timer import TimerCUDA

from corallab_lib import MotionPlanningProblem

from corallab_planners import Planner


PROJECT_ROOT = "/home/tassos/phd/qureshi/first-project/wip_trajectory_generator"
PLANNER_SOLUTION_CACHE = os.path.join(PROJECT_ROOT, "solution_cache")


class CachePlanner:

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            cached_planner_name : str = None,
            cached_planner_backend : str = None,
            cached_planner_config : dict = None,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        self.problem = problem
        self.tensor_args = tensor_args

        # Planner
        self.cached_planner_name = cached_planner_name
        self.cached_planner_backend = cached_planner_backend
        self.cached_planner_config = cached_planner_config

        self.cached_planner = Planner(
            cached_planner_name,
            problem=problem,
            backend=cached_planner_backend,
            **cached_planner_config
        )

    @property
    def name(self):
        return f"{self.cached_planner.name}_cache"

    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs
    ):
        """Use cached solution for current problem instance and planner. If not
        found, call planner.

        """

        start_hash = hash(tuple(start.tolist()))
        goal_hash = hash(tuple(goal.tolist()))

        current_cache_dir = os.path.join(
            PLANNER_SOLUTION_CACHE,
            self.cached_planner.name,
            self.problem.env.name,
            self.problem.robot.name,
            str(start_hash + goal_hash)
        )

        if os.path.exists(current_cache_dir):
            solution = torch.load(
                os.path.join(current_cache_dir, 'solution.pt'),
                map_location=self.tensor_args['device']
            )

            print(f"Read solution from cache {current_cache_dir}")

            try:
                with open(os.path.join(current_cache_dir, "info.pkl"), "rb") as info_file:
                    info = pickle.load(info_file)
                print(f"Read info from cache")
            except FileNotFoundError:
                print(f"Couldn't find cached info")
                info = {}


            if solution.ndim == 2:
                solution = solution.unsqueeze(0)

            if solution.shape[0] >= n_trajectories:
                solution = solution[:n_trajectories]

                # Not caching info for now...
                return solution, info

        # Otherwise, plan...

        with TimerCUDA() as planner_solve_timer:
            solution, info = self.cached_planner.solve(
                start, goal, n_trajectories=n_trajectories
            )

        t_total = planner_solve_timer.elapsed

        info["cached_solve_time"] = t_total

        if isinstance(solution, list):
            solution = np.stack(solution)

        if solution is None or info.get("failed", False):
            print("No solution")
            failed = True
            solution = torch.empty(0, 0, 0)
        else:
            failed = False
            solution = torch.tensor(solution, **self.tensor_args)

        if solution.ndim == 2:
            solution = solution.unsqueeze(0)

        print(f"saving solution with shape: {solution.shape}")

        Path(current_cache_dir).mkdir(parents=True, exist_ok=True)
        torch.save(solution, os.path.join(current_cache_dir, f'solution.pt'))

        print(f"Wrote solution to cache {current_cache_dir}")

        with open(os.path.join(current_cache_dir, "info.pkl"), "wb") as info_file:
            pickle.dump(info, info_file)

        print(f"Wrote info to cache")

        return solution, info

    def render(self, ax, **kwargs):
        pass
