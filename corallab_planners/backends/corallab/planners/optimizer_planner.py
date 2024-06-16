from corallab_lib.task import Task

from corallab_planners import Planner, Optimizer


class OptimizerPlanner:
    def __init__(
            self,
            task : Task = None,

            initial_planner : Planner = None,
            initial_planner_name : str = "RRTConnect",
            initial_planner_backend = "ompl",
            initial_planner_args : dict = {},

            optimizer : Optimizer = None,
            optimizer_name : str = "CHOMP",
            optimizer_backend = "mp_baselines",
            optimizer_args : dict = {},
            **kwargs
    ):
        self.planner = Planner(
            planner_name = initial_planner_name,
            task = task,
            **initial_planner_args,
            backend = initial_planner_backend,
        )

        self.optimizer = Optimizer(
            optimizer_name = optimizer_name,
            task = task,
            **optimizer_args,
            backend = optimizer_backend,
        )

    def solve(
            self,
            start,
            goal,
            # return_iterations=True,
            **kwargs
    ):
        initial_solution = self.planner.solve(start, goal, **kwargs)
        refined_solution = self.optimizer.optimize(guess=initial_solution)
        return refined_solution
