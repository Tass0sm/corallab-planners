from .entity import Entity


class DynamicPlanner(Entity):
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        super().__init__("DynamicPlannerImpl", *args, backend=backend, from_impl=from_impl, **kwargs)
