import sys
import math
import torch
import numpy as np

from corallab_lib import MotionPlanningProblem
from corallab_lib.backends.torch_robotics.env_impl import TorchRoboticsEnv
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

from corallab_lib import MotionPlanningProblem
from corallab_planners import Planner


class MultiAgentPRMPlanner:
    """
    Superclass for planners that construct a roadmaps for each unique agent.
    """

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            **kwargs
    ):
        assert problem.robot.is_multi_agent()

        self.problem = problem
        self.subrobots = self.problem.robot.get_subrobots()
        self.n_subrobots = len(self.subrobots)
        self.unique_subrobots = []
        self.subrobot_to_unique_subrobot_map = {}

        j = 0
        for i, r in enumerate(self.subrobots):
            if r not in self.unique_subrobots:
                self.unique_subrobots.append(r)
                self.subrobot_to_unique_subrobot_map[i] = j
                j += 1
            else:
                unique_idx = self.unique_subrobots.index(r)
                self.subrobot_to_unique_subrobot_map[i] = unique_idx

        self.subrobot_dict = {}
        self.prms = []

        # Subrobot PRM planners
        for r in self.unique_subrobots:
            backend = "torch_robotics" if isinstance(problem.env, TorchRoboticsEnv) else "curobo"

            single_robot_problem = MotionPlanningProblem(
                env_impl=problem.env,
                robot=r,
                backend=backend
            )

            prm = Planner(
                "PRM",
                problem=single_robot_problem,
                backend="corallab"
            )

            self.subrobot_dict[i] = (single_robot_problem, prm)
            self.prms.append(prm)

    def _get_subrobot_states(self, joint_state):
        states = []

        for i, r in enumerate(self.subrobots):
            subrobot_state = r.get_position(joint_state)
            states.append(subrobot_state)
            joint_state = joint_state[r.get_n_dof():]

        return states

    def _create_joint_solution(self, solutions, concatenate=False):
        solutions_l = list(solutions.values())
        max_length = max([sol.shape[0] for sol in solutions_l])

        def pad(t):
            remainder =  max_length - t.shape[0]
            last_state = t[-1]
            padding = last_state.repeat((remainder, 1))
            return torch.cat((t, padding))

        padded_solutions_l = list(map(pad, solutions_l))

        if concatenate:
            joint_solution = torch.cat(padded_solutions_l, axis=-1)
        else:
            joint_solution = padded_solutions_l

        return joint_solution

    def _visualize_roadmap(self):
        import torch
        from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

        tr_s_task = self.prms[0].planner_impl.planner_impl.problem.task_impl
        tr_visualizer = PlanningVisualizer(
            task=tr_s_task,
            planner=self.prms[0].planner_impl.planner_impl
        )

        # visualize trajectories in environment
        fig, axs = tr_visualizer.render_robot_trajectories(
            trajs=torch.zeros((1, 64, 2), **self.prms[0].planner_impl.planner_impl.tensor_args),
            render_planner=True
        )

        fig.show()

    def solve(
            self,
            start,
            goal,
            **kwargs
    ):
        """Abstract solve function.
        """
        raise NotImplementedError
