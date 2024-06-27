import torch
import numpy as np

from corallab_lib.task import Task

from . import planners
from ..planner_interface import PlannerInterface


class CorallabPlanner(PlannerInterface):

    def __init__(
            self,
            planner_name : str,
            task : Task = None,
            **kwargs
    ):
        PlannerClass = getattr(planners, planner_name)
        self.planner_impl = PlannerClass(
            task=task,
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
