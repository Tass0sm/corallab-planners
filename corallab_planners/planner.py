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
            self.planner_impl = PlannerImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.planner_impl = PlannerImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.planner_impl, name):
            return getattr(self.planner_impl, name)
        else:
            # Default behaviour
            raise AttributeError
