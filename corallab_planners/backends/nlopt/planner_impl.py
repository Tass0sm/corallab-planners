import time
import torch
import torch.func
from typing import Callable

import einops
import mlflow
from math import ceil
from abc import abstractmethod

import numpy as np


from corallab_lib import MotionPlanningProblem
from corallab_planners.backends.planner_interface import PlannerInterface

from torch_robotics.torch_utils.torch_utils import (
    DEFAULT_TENSOR_ARGS, freeze_torch_model_params, tensor_linspace_v1
)

import scipy.interpolate

import nlopt


class NLoptPlanner(PlannerInterface):
    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,

            n_segments : int = None,
            min_n_segments : int = 1,
            max_n_segments : int = None,

            **kwargs
    ):
        self.problem = problem

        if n_segments is None:
            assert max_n_segments is not None
            self.min_n_segments = min_n_segments
            self.max_n_segments = max_n_segments
        else:
            self.min_n_segments = n_segments
            self.max_n_segments = n_segments

        self.t_max : int = 64
        self.v_max = 0.2

    def _view_x_as_path(self, x):
        dim = self.problem.get_q_dim()
        return x.reshape(-1, dim, order="C")

    def _add_initial_state_constraint(self, opt, start):
        start_state = start.cpu()

        def initial_state_cs(result, x, grad):
            path = self._view_x_as_path(x)
            path = torch.tensor(path, requires_grad=True)

            def get_initial_state_cs(path):
                cs = path[0] - start_state
                return cs, cs

            jacobian, cs = torch.func.jacrev(get_initial_state_cs, has_aux=True)(path)

            if grad.size > 0:
                grad[:] = jacobian.detach().flatten(1).numpy()

            result[:] = cs.detach().numpy()

        tol = np.array([1e-3] * self.problem.get_q_dim())
        opt.add_equality_mconstraint(initial_state_cs, tol)

    def _add_goal_state_constraint(self, opt, goal):
        goal = goal.cpu()

        def goal_state_cs(result, x, grad):
            path = self._view_x_as_path(x)
            path = torch.tensor(path, requires_grad=True)

            def get_goal_state_cs(path):
                cs = path[-1] - goal
                return cs, cs

            jacobian, cs = torch.func.jacrev(get_goal_state_cs, has_aux=True)(path)

            if grad.size > 0:
                grad[:] = jacobian.detach().flatten(1).numpy()

            result[:] = cs.detach().numpy()

        tol = np.array([1e-3] * (self.problem.get_q_dim()))
        opt.add_equality_mconstraint(goal_state_cs, tol)

    def _add_space_constraints(self, opt, n_points):
        q_min = self.problem.get_q_min().repeat((n_points, 1)).cpu().numpy().flatten()
        q_max = self.problem.get_q_max().repeat((n_points, 1)).cpu().numpy().flatten()

        opt.set_lower_bounds(q_min)
        opt.set_upper_bounds(q_max)

    def solve(
            self,
            start,
            goal,
            objective=None,
            # stl_expression=None,
            # n_trajectories=1,
            **kwargs
    ):
        for n_segments in range(self.min_n_segments, self.max_n_segments + 1):
            n_points = n_segments + 1
            solution, info = self._get_n_point_solution(
                start,
                goal,
                n_points,
                objective=objective
            )
            break

        print("optimum at ", solution)
        print("minimum value = ", info["value"])
        print("num evaluations = ", info["num_evaluations"])
        print("result code = ", info["result_code"])

        return solution, info

    def _get_n_point_solution(
            self,
            start,
            goal,
            n_points : int,
            objective : Callable = None,
            **kwargs
    ):
        def nlopt_objective(x, grad):
            path = self._view_x_as_path(x)

            path = torch.tensor(path, requires_grad=True)
            paths = path.unsqueeze(0).float().cuda()
            obj = objective(paths).squeeze()
            obj.backward()

            if grad.size > 0:
                grad[:] = path.grad.flatten().cpu().numpy()

            return obj.detach().item()

        n_parameters = self.problem.get_q_dim() * n_points
        opt = nlopt.opt(nlopt.LD_AUGLAG, n_parameters)

        self._add_initial_state_constraint(opt, start)
        self._add_goal_state_constraint(opt, goal)
        self._add_space_constraints(opt, n_points)

        opt.set_min_objective(nlopt_objective)
        opt.set_ftol_abs(1e-4)
        opt.set_xtol_abs(1e-4)
        # opt.set_maxeval(6000000);
        # opt.set_maxtime(2.0);

        guess = tensor_linspace_v1(
            start, goal, steps=n_points
        ).T.flatten().cpu().numpy()

        solution = opt.optimize(guess)
        solution = self._view_x_as_path(solution)

        return solution, {
            "value": opt.last_optimum_value(),
            "num_evaluations": opt.get_numevals(),
            "result_code": opt.last_optimize_result(),
        }

    def reset(self):
        pass
