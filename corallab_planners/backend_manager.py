from .backends import backend_list as corallab_planners_backend_list
import importlib
import types

from corallab_lib.backend_manager import BackendManager


backend_manager = BackendManager("corallab_planners.backends", corallab_planners_backend_list)
