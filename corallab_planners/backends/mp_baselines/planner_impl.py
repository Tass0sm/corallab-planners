import torch
import numpy as np

from corallab_lib import MotionPlanningProblem
from corallab_lib.backends.torch_robotics.motion_planning_problem_impl import TorchRoboticsMotionPlanningProblem

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.mppi import MPPI
from mp_baselines.planners.rrt_connect import RRTConnect
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.stoch_gpmp import StochGPMP
from mp_baselines.planners.stomp import STOMP

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS

from ..planner_interface import PlannerInterface


mp_baselines_planners = {
    "CHOMP": CHOMP,
    "GPMP2": GPMP2,
    "MPPI": MPPI,
    "RRTConnect": RRTConnect,
    "RRTStar": RRTStar,
    "StochGPMP": StochGPMP,
    "STOMP": STOMP
}


class MPBaselinesPlanner(PlannerInterface):

    def __init__(
            self,
            planner_name : str,
            problem : MotionPlanningProblem = None,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        assert problem.backend == "torch_robotics", "MPBaselinesPlanner only supports problems using the torch_robotics backend."

        self.n_dof = problem.get_q_dim()
        self.task_impl = problem.problem_impl.task_impl
        self.robot_impl = problem.robot.robot_impl
        self.tensor_args = self.task_impl.tensor_args
        self.planner_name = planner_name
        self.PlannerClass = mp_baselines_planners[planner_name]
        self.config = kwargs

        if self.planner_name == "CHOMP":
            self.config["pos_only"] = False

    @property
    def name(self):
        return f"mp_baselines_{self.planner_impl.name}"

    @property
    def n_support_points(self):
        return self.planner_impl.n_support_points

    @property
    def dt(self):
        return self.planner_impl.dt

    def solve(
            self,
            start,
            goal,
            objective=None,
            **kwargs
    ):
        start_state = start
        start_state_pos = start
        goal_state = goal
        goal_state_pos = goal
        multi_goal_states = goal.unsqueeze(0)

        planner_impl = self.PlannerClass(
            n_dof=self.n_dof,
            task=self.task_impl,
            robot=self.robot_impl,

            start_state=start_state,
            start_state_pos=start_state_pos,

            goal_state=goal_state,
            goal_state_pos=goal_state_pos,
            multi_goal_states=multi_goal_states,

            num_particles_per_goal=1,
            collision_fields=self.task_impl.get_collision_fields(),
            cost=objective,

            tensor_args=self.tensor_args,
            **self.config,
            **kwargs,
        )

        solution = planner_impl.optimize(**kwargs)
        info = {}

        if solution is None:
            return solution, info

        if solution.ndim == 3:
            return solution, info
        elif solution.ndim == 2:
            return solution.unsqueeze(0), info

    def reset(self):
        # self.planner_impl.reset()
        pass
