from .backend_manager import backend_manager

class Planner:
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        PlannerImpl = backend_manager.get_backend_attr(
            "PlannerImpl",
            backend=backend
        )

        if from_impl:
            self.robot_impl = PlannerImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.robot_impl = PlannerImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.robot_impl, name):
            return getattr(self.robot_impl, name)
        else:
            # Default behaviour
            raise AttributeError
