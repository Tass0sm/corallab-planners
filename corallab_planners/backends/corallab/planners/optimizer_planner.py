import torch

from corallab_lib.task import Task

from corallab_planners import Planner, Optimizer


class OptimizerPlanner:
    def __init__(
            self,
            task : Task = None,

            initial_planner : Planner = None,
            initial_planner_name : str = "StraightLinePlanner",
            initial_planner_backend = "corallab",
            initial_planner_args : dict = {},

            optimizer : Optimizer = None,
            optimizer_name : str = "CHOMP",
            optimizer_backend = "mp_baselines",
            optimizer_args : dict = {},
            **kwargs
    ):
        self.task = task

        self.initial_planner = Planner(
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

    @property
    def name(self):
        return f"{self.initial_planner.name}_and_{self.optimizer.name}"

    def solve(
            self,
            start,
            goal,
            objective=None,
            **kwargs
    ):

        initial_solution, planner_info = self.initial_planner.solve(start, goal, **kwargs)

        if initial_solution is None:
            return None, {}

        refined_solution, optimizer_info = self.optimizer.optimize(
            guess=initial_solution,
            objective=objective
        )

        if "solution_iters" in planner_info and "solution_iters" in optimizer_info:
            s_iters1 = self.task.robot.get_position(planner_info["solution_iters"])
            s_iters2 = self.task.robot.get_position(optimizer_info["solution_iters"])

            s_iters = torch.cat([s_iters1, s_iters2])

            info = { "solution_iters": s_iters }
        else:
            info = dict(**planner_info, **optimizer_info)

        return refined_solution, info
