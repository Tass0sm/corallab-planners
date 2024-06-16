import torch
import numpy as np

from corallab_lib.task import Task

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
            task : Task = None,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        q_dim = task.get_q_dim()
        tmp_start_state = torch.zeros((2 * q_dim,))
        tmp_start_state_pos = torch.zeros((q_dim,))
        tmp_goal_state = torch.zeros((2 * q_dim,))
        tmp_goal_state_pos = torch.zeros((q_dim,))

        task_impl = task.task_impl.task_impl

        PlannerClass = mp_baselines_planners[planner_name]

        self.planner_impl = PlannerClass(
            task=task_impl,

            start_state=tmp_start_state,
            start_state_pos=tmp_start_state_pos,

            goal_state=tmp_goal_state,
            goal_state_pos=tmp_goal_state_pos,
            multi_goal_states=tmp_goal_state.unsqueeze(0),

            tensor_args=tensor_args,
            **kwargs,
        )

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
            # return_iterations=True,
            **kwargs
    ):
        self.planner_impl.start_state_pos = start
        self.planner_impl.goal_state_pos = goal
        self.planner_impl.multi_goal_states = goal.unsqueeze(0)
        self.planner_impl.reset()

        solution = self.planner_impl.optimize(**kwargs)

        # TODO: Decide on API
        return solution.unsqueeze(0).unsqueeze(0)

    def reset(self):
        self.planner_impl.reset()
