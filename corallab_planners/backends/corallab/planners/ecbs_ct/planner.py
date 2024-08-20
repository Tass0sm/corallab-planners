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

from corallab_planners.backends.corallab.planners.prm.utils import constrained_a_star

from typing import Callable

from corallab_lib import MotionPlanningProblem

import scipy.interpolate

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.torch_utils.torch_timer import TimerCUDA



import numpy as np
import pylab as pl
import time

# from .tree import Tree
# from .implicit_graph import ImplicitGraph
from corallab_lib import MotionPlanningProblem
from corallab_planners import Planner

from collections import defaultdict

import random

from ..multi_agent_prm_planner import MultiAgentPRMPlanner


class Constraint:

    def __init__(self, i, traj, time):
        self.i = i
        self.traj = traj
        self.time = time


class Conflict:

    def __init__(self, i, j, traj_i, traj_j, time):
        self.i = i
        self.j = j
        self.traj_i = traj_i
        self.traj_j = traj_j
        self.time = time

    # def __str__(self):
    #     return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
    #         "EC: " + str([str(ec) for ec in self.edge_constraints])


class HighLevelNode:

    def __init__(self, solutions, constraints=set(), parent=None):
        self.solutions = solutions
        self.constraints = constraints
        self.parent = parent
        self.compute_cost()

    def set_plan(self, solutions, times):
        self.solutions = solutions
        self.times = times

    def has_full_solution(self):
        return all([sol is not None for sol in self.solutions.values()])

    def compute_cost(self):
        if self.solutions:
            self.cost = 0

            for sol in self.solutions.values():
                if sol is None:
                    self.cost += 999
                else:
                    self.cost += sol.diff(dim=0).square().sum(dim=-1).sqrt().sum()
        else:
            self.cost = 999.0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost


class ECBS_CT(MultiAgentPRMPlanner):
    """Simple implementation of:

    Cohen, Uras & Kumar et al. (2019) Optimal and Bounded-Suboptimal Multi-Agent
    Motion Planning, Proceedings of the International Symposium on Combinatorial
    Search.
    """

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            w : float = 1.0,
            n_iters: int = 30000,
            max_time: float = 60.,
            visualize = False,
            **kwargs
    ):
        super().__init__(problem=problem)

        self.n_iters = n_iters
        self.max_time = max_time

        self.w = w
        self.task = task
        self.implicit_graph = None
        self.visualize = visualize

    def _distance_fn(self, q1, q2):
        return self.task.distance_q(q1, q2).item()

    def _collision_fn(self, qs, **observation):
        return self.task.check_collision(qs).squeeze()

    # TODO:

    def _solve_individual_problems(self, start, goal):
        # Add start and goal positions
        subrobot_starts = self._get_subrobot_states(start)
        subrobot_goals = self._get_subrobot_states(goal)

        sols = {}

        for i, (start_i, goal_i, robot) in enumerate(zip(subrobot_starts, subrobot_goals, self.subrobots)):
            unique_subrobot_idx = self.subrobot_to_unique_subrobot_map[i]
            prm = self.prms[unique_subrobot_idx]

            sol, info = prm.solve(start_i, goal_i)

            sols[i] = sol

        return sols

    def _get_first_conflict(self, node):
        solutions = node.solutions

        joint_solutions_l = self._create_joint_solution(solutions, concatenate=False)
        joint_solution = torch.cat(joint_solutions_l, dim=-1)

        r_i, r_j = None, None
        t_s, t_e = None, None
        idx_s, idx_e = None, None

        i = 0
        while i < len(joint_solution):
            state = joint_solution[i]
            in_collision, info = self.task.check_collision_info(state, margin=0)

            assert not info["cost_collision_objects"], "Somehow found object collision?"
            assert not info["cost_collision_border"], "Somehow found border collision?"

            if in_collision:
                idx_s = i

                r_i, r_j = info["self_collision_robots"][0, 1:]
                r_i = r_i.item()
                r_j = r_j.item()

                j = i + 1
                while j < len(joint_solution):
                    state = joint_solution[j]
                    in_collision, info = self.task.check_collision_info(state, margin=0)

                    if info["self_collision_robots"].nelement() == 0:
                        self_collision_robots = []
                    else:
                        self_collision_robots = info["self_collision_robots"][:, 1:].flatten().unique()

                    if r_i not in self_collision_robots and r_j not in self_collision_robots:
                        idx_e = j+1
                        break

                    j += 1

            if t_e is not None:
                break

            i += 1

        if r_i is None:
            return None
        else:
            time = torch.arange(idx_s, idx_e+1)
            traj_i = joint_solutions_l[r_i][idx_s:idx_e+1]
            traj_j = joint_solutions_l[r_j][idx_s:idx_e+1]

            return Conflict(r_i, r_j, traj_i, traj_j, time)

    def _attempt_replan(self, robot_idx, node, start, goal):
        unique_subrobot_idx = self.subrobot_to_unique_subrobot_map[robot_idx]
        prm = self.prms[unique_subrobot_idx]

        tmp = node
        constraint_set = set()
        while tmp is not None:
            constraint_set |= tmp.constraints
            tmp = tmp.parent

        # Solve new dynamic problem
        subrobot_starts = self._get_subrobot_states(start)
        subrobot_goals = self._get_subrobot_states(goal)
        start_i = subrobot_starts[robot_idx]
        goal_i = subrobot_goals[robot_idx]

        # TODO: How to make this work??????
        sol, info = constrained_a_star(prm.planner_impl.planner_impl, start_i, goal_i, constraint_set)

        if isinstance(sol, list):
            sol = sol[0]

        if sol is not None:
            node.solutions[robot_idx] = sol
            node.times[robot_idx] = time
        else:
            print("Failed Replanning")
            node.solutions[robot_idx] = None

        node.compute_cost()

    def _scipp(
            self,
            prm,
            start,
            goal,
            constraints,
            reservation_table,
            max_time=30,
            n_iters=10000
    ):

        prm.grow_roadmap_with_samples([start, goal])

        open_s = set(prm.roadmap[start])
        focal_s = set()

        iteration = -1
        solution = None

        with TimerCUDA() as t:
            while (t.elapsed < max_time) and (iteration < n_iters) and focal != {}:
                iteration += 1

        return

        pass

    def solve(
            self,
            start,
            goal,
            prm_construction_time : float = 10.0,
            **kwargs
    ):
        start = start.to(**self.task.tensor_args)
        goal = goal.to(**self.task.tensor_args)

        # Create subrobot PRM
        for prm in self.prms:
            print("Constructing a PRM...")
            prm.planner_impl.planner_impl.construct_roadmap(
                allowed_time=prm_construction_time
            )

        # Solve individual problems
        solutions = self._solve_individual_problems(start, goal)
        ct_root = HighLevelNode(solutions)
        queue = [ct_root]

        iteration = -1
        solution = None

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                current_node = queue[0]

                if not current_node.has_full_solution():
                    current_node = heapq.heappop(queue)
                    solutions, times = self._solve_individual_problems(start, goal)
                    current_node.set_plan(solutions, times)

                    heapq.heappush(queue, current_node)
                else:
                    k = self._get_first_conflict(current_node)

                    if k is None:
                        solution = current_node;
                        break;
                    else:
                        current_node = heapq.heappop(queue)

                        for a_idx, a_traj, b_idx in [(k.i, k.traj_i, k.j),
                                                     (k.j, k.traj_j, k.i)]:
                            new_node_constraints = {Constraint(a_idx, a_traj, k.time)}
                            new_node = HighLevelNode(
                                copy(current_node.solutions),
                                constraints=new_node_constraints,
                                parent=current_node
                            )

                            self._attempt_replan(b_idx, new_node, start, goal)
                            heapq.heappush(queue, new_node)
