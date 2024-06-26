import torch
import numpy as np

from corallab_lib.task import Task

from . import optimizers
from ..optimizer_interface import OptimizerInterface


class CorallabOptimizer(OptimizerInterface):

    def __init__(
            self,
            optimizer_name : str,
            task : Task = None,
            **kwargs
    ):
        OptimizerClass = getattr(optimizers, optimizer_name)
        self.optimizer_impl = OptimizerClass(
            task=task,
            **kwargs
        )

    @property
    def name(self):
        return f"corallab_{self.optimizer_impl.name}"

    def optimize(self, **kwargs):
        return self.optimizer_impl.optimize(**kwargs)
