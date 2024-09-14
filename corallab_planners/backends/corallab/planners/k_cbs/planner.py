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
from corallab_planners import DynamicPlanner

from typing import Callable

from corallab_lib import MotionPlanningProblem

from corallab_lib.backends.torch_robotics.env_impl import TorchRoboticsEnv
from corallab_lib.backends.torch_robotics.motion_planning_problem_impl import TorchRoboticsMotionPlanningProblem
from corallab_lib.backends.pybullet.env_impl import PybulletEnv

import scipy.interpolate

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.robots import MultiRobot


class Constraint:

    def __init__(self, i, traj, time):
        self.i = i
        self.traj = traj
        self.time = time

    def __str__(self):
        return f"Constrait with {self.i} in {self.traj} over {self.time}"


class Conflict:

    def __init__(self, i, j, traj_i, traj_j, time):
        self.i = i
        self.j = j
        self.traj_i = traj_i
        self.traj_j = traj_j
        self.time = time

    def __str__(self):
        return f"Conflict over {self.time}"

    # def __str__(self):
    #     return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
    #         "EC: " + str([str(ec) for ec in self.edge_constraints])

class HighLevelNode:

    def __init__(self, solutions, times, constraints=set(), parent=None):
        self.solutions = solutions
        self.times = times
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
                    self.cost += np.linalg.norm(np.diff(sol, axis=0), axis=-1).sum()
                    # self.cost += sol.diff(dim=0).square().sum(dim=-1).sqrt().sum()
        else:
            self.cost = 999.0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost

    def __str__(self):
        return f"""HighLevelNode
constraints = {self.constraints}
cost = {self.cost}"""


class K_CBS:

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            n_iters: int = 30000,
            max_time: float = 60.,
            merge_threshold : int = 10,
            interpolate_solution: bool = True,
            interpolate_num: int = 64,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        """
        merge_threshold: The threshold on the number of conflicts between
        two robots before they are replanned jointly.
        """

        assert problem.robot.is_multi_agent()

        self.n_iters = n_iters
        self.max_time = max_time
        self.merge_threshold = merge_threshold

        self.interpolate_solution = interpolate_solution
        self.interpolate_num = interpolate_num

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

            single_robot_problem = DynamicPlanningProblem(
                env_impl=env_impl,
                robot=r,
                backend=backend
            )

            self.subrobot_problems.append(single_robot_problem)

            single_robot_planner = DynamicPlanner(
                planner_name="RRT",
                problem=single_robot_problem,
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
        sol_times = {}

        for i, ((problem, planner), start, goal) in enumerate(zip(self.subrobot_dict.values(), separate_starts, separate_goals)):
            sol, info = planner.solve(start, goal)

            if sol is None:
                sols[i] = None
                sol_times[i] = None
            else:
                time = info["time"]

                if isinstance(sol, list):
                    sol = sol[0]

                if isinstance(time, list):
                    time = time[0]

                sols[i] = sol
                sol_times[i] = time

        return sols, sol_times

    def _interpolate_times(self, times):
        request_count = self.interpolate_num
        times_l = times.tolist()

        if request_count < len(times) or len(times) < 2:
            return

        count = request_count;

        # the remaining length of the path we need to add states along
        remaining_length = times_l[-1] - times_l[0]

        # the new array of states this path will have
        new_times = []
        n1 = len(times_l) - 1;

        for i in range(n1):
            t1 = times_l[i];
            t2 = times_l[i + 1];

            new_times.append(t1)

            # the maximum number of states that can be added on the current motion (without its endpoints)
            # such that we can at least fit the remaining states
            max_n_states = count + i - len(times_l)

            if max_n_states > 0:
                # compute an approximate number of states the following segment needs to contain; this includes endpoints
                segment_length = t2 - t1
                ns = max_n_states + 2 if i + 1 == n1 else math.floor(0.5 + count * segment_length / remaining_length) + 1;

                if ns > 2:
                    if (ns - 2 > max_n_states):
                        ns = max_n_states + 2

                    segment_times = torch.linspace(t1, t2, ns).tolist()
                    new_times.extend(segment_times[1:-1])

                    ns -= 2
                else:
                    ns = 0

                # update what remains to be done
                count -= (ns + 1)
                remaining_length -= segment_length
            else:
                count -= 1

        # add the last state
        new_times.append(times_l[n1])
        times = torch.tensor(new_times)

        return times

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

    def _get_first_conflict(self, node):
        times = node.times
        solutions = node.solutions
        combined_times, joint_solutions_l = self._create_joint_solution(times, solutions, self.interpolate_num * 5, concatenate=False)
        joint_solution = np.concatenate(joint_solutions_l, axis=-1)

        r_i, r_j = None, None
        t_s, t_e = None, None
        idx_s, idx_e = None, None

        i = 0
        while i < len(joint_solution):
            state = joint_solution[i].reshape(1, 1, -1)
            # in_collision, info = self.task.get_self_collision_info(state)
            in_collision, info = self.problem.compute_collision_info(state, margin=0)

            if info["cost_collision_objects"]:
                print("Somehow found object collision?")
                break

            if info["cost_collision_border"]:
                print("Somehow found border collision?")
                break

            if in_collision:
                t_s = combined_times[i]
                idx_s = i

                r_i, r_j = info["self_collision_robots"][0, 1:]
                r_i = r_i.item()
                r_j = r_j.item()

                idx_e = i + 1
                while idx_e < len(joint_solution):
                    state = joint_solution[idx_e].reshape(1, 1, -1)
                    in_collision, info = self.problem.compute_collision_info(state, margin=0)

                    if info["self_collision_robots"] is None or info["self_collision_robots"].nelement() == 0:
                        self_collision_robots = []
                    else:
                        self_collision_robots = info["self_collision_robots"][:, 1:].flatten().unique()

                    if r_i not in self_collision_robots and r_j not in self_collision_robots:
                        t_e = combined_times[idx_e]
                        break

                    idx_e += 1

            if t_e is not None:
                break

            i += 1

        if r_i is None:
            return None
        else:
            time = combined_times[idx_s:idx_e+1]
            traj_i = joint_solutions_l[r_i][idx_s:idx_e+1]
            traj_j = joint_solutions_l[r_j][idx_s:idx_e+1]

            return Conflict(r_i, r_j, traj_i, traj_j, time)

    def _merge_subrobots(self, i, j):
        assert isinstance(self.problem.problem_impl, TorchRoboticsMotionPlanningProblem), "Merging is only supported for the torch_robotics backend"

        tr_multi_robot = MultiRobot(
            subrobots=[self.subrobots[i].robot_impl.robot_impl,
                       self.subrobots[j].robot_impl.robot_impl],
            tensor_args=self.tensor_args,
        )

        merged_robot = Robot(
            from_impl=tr_multi_robot,
            backend="torch_robotics"
        )

        merged_robot_problem = DynamicPlanningProblem(
            env_impl=self.problem.env,
            robot=merged_robot,
            backend="torch_robotics"
        )

        merged_robot_planner = DynamicPlanner(
            planner_name="RRT",
            problem=merged_robot_problem,
            backend="ompl"
        )

        del self.subrobot_dict[i]
        del self.subrobot_dict[j]

        self.subrobot_dict[i] = (merged_robot_problem, merged_robot_planner)
        self.subrobot_dict[j] = (merged_robot_problem, merged_robot_planner)

    def _attempt_replan(self, robot_idx, node, start, goal):

        problem, planner = self.subrobot_dict[robot_idx]
        problem.clear_dynamic_obstacles()

        tmp = node
        constraint_set = set()
        while tmp is not None:
            constraint_set |= tmp.constraints
            tmp = tmp.parent

        for c in constraint_set:
            c_rob = self.subrobots[c.i].robot_impl
            problem.add_dynamic_obstacle(c_rob, c.traj, c.time)

        # Solve new dynamic problem
        separate_starts = self._separate_joint_state(start)
        separate_goals = self._separate_joint_state(goal)
        start_i = separate_starts[robot_idx]
        goal_i = separate_goals[robot_idx]


        sol, info = planner.solve(start_i, goal_i)
        time = info["time"]

        if isinstance(sol, list):
            sol = sol[0]

        if isinstance(time, list):
            time = time[0]

        if sol is not None:
            node.solutions[robot_idx] = sol
            node.times[robot_idx] = time
        else:
            print("Failed Replanning")
            node.solutions[robot_idx] = None
            node.times[robot_idx] = None

        node.compute_cost()

    def _vis_helper(self, node):
        times = node.times
        solutions = node.solutions
        combined_times, joint_solutions_l = self._create_joint_solution(times, solutions, concatenate=False)
        joint_solution = np.concatenate(joint_solutions_l, axis=-1)

        tr_task = self.problem.problem_impl.task_impl
        tr_visualizer = PlanningVisualizer(task=tr_task)
        fig, axs = tr_visualizer.render_robot_trajectories(trajs=torch.tensor(np.expand_dims(joint_solution, 0), **tr_task.tensor_args), start_state=joint_solution[0], goal_state=joint_solution[-1])
        fig.show()

    @property
    def name(self):
        return f"k_cbs"

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
        info_l = list(filter(lambda x: "joint_times" in x, info_l))

        if len(sol_l) == 0 or len(info_l) == 0:
            return None, {}

        joint_times = np.stack([info["joint_times"] for info in info_l])
        joint_solutions = np.stack(sol_l)

        return joint_solutions, { "joint_times": joint_times }

    def _get_single_solution(
            self,
            start,
            goal
    ):
        iteration = -1
        solution = None

        # create root node of constraint tree with an initial path for every individual
        solutions, times = self._solve_individual_problems(start, goal)
        ct_root = HighLevelNode(solutions, times)

        print("MADE INDIVIDUAL SOLUTIONS")

        conflict_count_matrix = np.zeros((self.n_subrobots, self.n_subrobots))
        queue = [ct_root]

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                current_node = queue[0]

                print(f"NOW LOOKING AT {current_node}")

                if not current_node.has_full_solution():
                    current_node = heapq.heappop(queue)
                    solutions, times = self._solve_individual_problems(start, goal)
                    current_node.set_plan(solutions, times)

                    heapq.heappush(queue, current_node)
                else:
                    k = self._get_first_conflict(current_node)

                    if k is None:
                        print(f"FOUND NO CONFLICT")
                        solution = current_node;
                        break;
                    elif conflict_count_matrix[k.i, k.j] > self.merge_threshold:
                        self._merge_subrobots(k.i, k.j)

                        solutions, times = self._solve_individual_problems(start, goal)
                        ct_root = HighLevelNode(solutions, times)

                        print("MADE INDIVIDUAL SOLUTIONS")

                        conflict_count_matrix = np.zeros((self.n_subrobots, self.n_subrobots))
                        queue = [ct_root]

                        breakpoint()
                    else:
                        print(f"FOUND CONFLICT {k}")

                        current_node = heapq.heappop(queue)

                        conflict_count_matrix[k.i, k.j] += 1
                        conflict_count_matrix[k.j, k.i] += 1

                        for a_idx, a_traj, b_idx in [(k.i, k.traj_i, k.j),
                                                     (k.j, k.traj_j, k.i)]:
                            new_node_constraints = {Constraint(a_idx, a_traj, k.time)}
                            new_node = HighLevelNode(
                                copy(current_node.solutions),
                                copy(current_node.times),
                                constraints=new_node_constraints,
                                parent=current_node
                            )

                            print(f"REPLANNING FOR {b_idx}")

                            self._attempt_replan(b_idx, new_node, start, goal)
                            heapq.heappush(queue, new_node)

                        # solution = new_node
                        # break


        if solution is None:
            print("None Found!!!!!!!")
            return None, {}

        joint_times, joint_solution = self._create_joint_solution(solution.times, solution.solutions, self.interpolate_num, concatenate=True)

        return joint_solution, { "joint_times": joint_times }

    def distance_fn(self, q1, q2):
        return self.problem.distance_q(q1, q2).item()

    def collision_fn(self, qs, **observation):
        return self.problem.check_collision(qs).squeeze()

    def reset(self):
        pass
