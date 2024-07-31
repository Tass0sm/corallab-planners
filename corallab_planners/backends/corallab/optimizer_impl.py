import torch
import numpy as np

from corallab_lib import MotionPlanningProblem

from . import optimizers
from ..optimizer_interface import OptimizerInterface


class CorallabOptimizer(OptimizerInterface):

    def __init__(
            self,
            optimizer_name : str,
            problem : MotionPlanningProblem = None,
            **kwargs
    ):
        OptimizerClass = getattr(optimizers, optimizer_name)
        self.optimizer_impl = OptimizerClass(
            problem=problem,
            **kwargs
        )

    @property
    def name(self):
        return f"corallab_{self.optimizer_impl.name}"

    def optimize(self, **kwargs):
        return self.optimizer_impl.optimize(**kwargs)
