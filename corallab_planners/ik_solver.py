from .backend_manager import backend_manager

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


class IKSolver:
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        IKSolverImpl = backend_manager.get_backend_attr(
            "IKSolverImpl",
            backend=backend
        )

        if from_impl:
            self.ik_solver_impl = IKSolverImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.ik_solver_impl = IKSolverImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.ik_solver_impl, name):
            return getattr(self.ik_solver_impl, name)
        else:
            # Default behaviour
            raise AttributeError
