from .backend_manager import backend_manager as corallab_planners_backend_manager

from corallab_lib.entity import BaseEntity


class Entity(BaseEntity):
    def __init__(
            self,
            entity_impl_name,
            *args,
            **kwargs,
    ):
        super().__init__(
            entity_impl_name,
            corallab_planners_backend_manager,
            *args,
            **kwargs
        )
