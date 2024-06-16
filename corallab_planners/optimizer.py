from .backend_manager import backend_manager

class Optimizer:
    def __init__(
            self,
            *args,
            backend=None,
            from_impl=None,
            **kwargs
    ):
        OptimizerImpl = backend_manager.get_backend_attr(
            "OptimizerImpl",
            backend=backend
        )

        if from_impl:
            self.optimizer_impl = OptimizerImpl.from_impl(from_impl, *args, **kwargs)
        else:
            self.optimizer_impl = OptimizerImpl(*args, **kwargs)

    def __getattr__(self, name):
        if hasattr(self.optimizer_impl, name):
            return getattr(self.optimizer_impl, name)
        else:
            # Default behaviour
            raise AttributeError
