import torch
import numpy as np

from corallab_lib import DynamicPlanningProblem

from . import dynamic_planners
# from ..planner_interface import PlannerInterface


class CorallabDynamicPlanner:

    def __init__(
            self,
            planner_name : str,
            problem : DynamicPlanningProblem = None,
            **kwargs
    ):
        DynamicPlannerClass = getattr(dynamic_planners, planner_name)
        self.planner_impl = DynamicPlannerClass(
            problem=problem,
            **kwargs
        )

    @property
    def name(self):
        return f"corallab_{self.planner_impl.name}"

    def solve(
            self,
            start,
            goal,
            **kwargs
    ):
        return self.planner_impl.solve(start, goal, **kwargs)

    def reset(self):
        pass
