from corallab_lib.task import Task

from corallab_planners import Planner, Optimizer


class RepeatPlanner:
    def __init__(
            self,
            task : Task = None,

            sub_planner : Planner = None,
            sub_planner_name : str = "RRTConnect",
            sub_planner_backend = "ompl",
            sub_planner_args : dict = {},

            **kwargs
    ):
        self.planner = Planner(
            planner_name = sub_planner_name,
            task = task,
            **sub_planner_args,
            backend = sub_planner_backend,
        )


    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs
    ):

        trajs_l = []
        for _ in range(n_trajectories):  # queue up multiple tasks
            traj, _ = self.planner.solve(start, goal)
            trajs_l.append(traj)

        return trajs_l, {}
