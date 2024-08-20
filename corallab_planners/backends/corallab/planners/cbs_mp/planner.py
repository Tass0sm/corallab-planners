import torch

from corallab_lib import MotionPlanningProblem, PebbleMotionProblem
from corallab_lib.backends.corallab import PebbleMotionContinuousValidator
from corallab_planners import Planner

from ..multi_agent_prm_planner import MultiAgentPRMPlanner


class CBS_MP(MultiAgentPRMPlanner):
    """
    Simple implementation of:
    "Representation-Optimal Multi-Robot Motion Planning Using Conflict-Based
    Search." IEEE Robotics and Automation Letters
    """

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            n_iters: int = 30000,
            max_time: float = 60.,
            interpolate_solution: bool = True,
            interpolate_num: int = 64,
            **kwargs
    ):
        super().__init__(problem=problem)

        self.n_iters = n_iters
        self.max_time = max_time
        self.interpolate_solution = interpolate_solution
        self.interpolate_num = interpolate_num

    def _create_joint_solution(self, solutions):
        solutions_l = list(solutions.values())
        max_length = max([sol.shape[0] for sol in solutions_l])

        def pad(t):
            remainder =  max_length - t.shape[0]
            last_state = t[-1]
            padding = last_state.repeat((remainder, 1))
            return torch.cat((t, padding))

        padded_solutions_l = list(map(pad, solutions_l))
        joint_solution = torch.cat(padded_solutions_l, dim=-1)
        return joint_solution

    def solve(
            self,
            start,
            goal,
            prm_construction_iters : int = 1,
            **kwargs
    ):
        assert len(self.prms) == 1, "CBS_MP only works for agents of a single type"

        self.construct_roadmaps()

        # self._visualize_roadmap()
        # breakpoint()

        # Add start and goal positions
        subrobot_starts = self._add_subrobot_states_to_prm(start)
        subrobot_goals = self._add_subrobot_states_to_prm(goal)

        # Construct Discrete Problem
        prm = self.prms[0]
        roadmap = prm.planner_impl.planner_impl.roadmap

        p = PebbleMotionProblem(
            graph=roadmap,
            validator=PebbleMotionContinuousValidator(self.problem),
            n_pebbles=self.n_subrobots,
            backend="corallab"
        )

        planner = Planner(
            "CBS",
            problem=p,
            n_iters=self.n_iters,
            max_time=self.max_time,
            backend="corallab"
        )

        print("Running CBS...")
        solutions, info = planner.solve(subrobot_starts, subrobot_goals)
        joint_solution = self._create_joint_solution(solutions)
        return joint_solution, info
