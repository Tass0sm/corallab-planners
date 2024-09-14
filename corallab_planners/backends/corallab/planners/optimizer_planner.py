import torch

from corallab_lib import MotionPlanningProblem

from corallab_planners import Planner, Optimizer


class OptimizerPlanner:
    def __init__(
            self,
            problem : MotionPlanningProblem = None,

            initial_planner : Planner = None,
            initial_planner_name : str = "StraightLinePlanner",
            initial_planner_backend = "corallab",
            initial_planner_args : dict = {},

            optimizer : Optimizer = None,
            optimizer_name : str = None, # "CHOMP",
            optimizer_backend = "mp_baselines",
            optimizer_args : dict = {},
            **kwargs
    ):
        self.problem = problem

        self.initial_planner = Planner(
            planner_name = initial_planner_name,
            problem = problem,
            **initial_planner_args,
            backend = initial_planner_backend,
        )

        self.optimizer = Optimizer(
            optimizer_name = optimizer_name,
            problem = problem,
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

        if initial_solution is None or initial_solution.nelement() == 0:
            return None, {}

        try:
            refined_solution, optimizer_info = self.optimizer.optimize(
                guess=initial_solution,
                objective=objective
            )
        except NotImplementedError:
            refined_solution, optimizer_info = None, {}

        if "solution_iters" in planner_info and "solution_iters" in optimizer_info:
            s_iters1 = self.problem.robot.get_position(planner_info["solution_iters"])
            s_iters2 = self.problem.robot.get_position(optimizer_info["solution_iters"])

            if s_iters1.shape[1:] == s_iters2.shape[2:]:
                s_iters = torch.cat([s_iters1, s_iters2])
                info = { "solution_iters": s_iters }
            else:
                info = {}
        else:
            info = dict(**planner_info, **optimizer_info)

        return refined_solution, info
