from corallab_lib.backends.corallab.implicit_graph import ImplicitGraph
from corallab_lib import MotionPlanningProblem, PebbleMotionProblem

from corallab_planners import Planner

from ..multi_agent_prm_planner import MultiAgentPRMPlanner


class M_STAR_MP(MultiAgentPRMPlanner):
    """Simple implementation of:

    Wagner & Choset (2011) M*: A Complete Multirobot Path Planning Algorithm
    with Performance Bounds, .
    """

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            n_iters: int = 30000,
            max_time: float = 60.,
            visualize = False,
            **kwargs
    ):
        super().__init__(problem=problem)

        self.n_iters = n_iters
        self.max_time = max_time

        self.collision_set_dict = None
        self.backprop_set_dict = None
        self.policy_dict = None

        self.implicit_graph = None
        self.explicit_roadmap = None

    def _distance_fn(self, q1, q2):
        return self.problem.distance_q(q1, q2).item()

    def _collision_fn(self, qs, **observation):
        return self.problem.check_collision(qs).squeeze()

    ##################################################

    def solve(
            self,
            start,
            goal,
            prm_construction_time : float = 5.0,
            **kwargs
    ):
        # Create subrobot PRM
        if not self.loaded_roadmaps:
            for prm in self.prms:
                print("Constructing a PRM...")
                prm.planner_impl.planner_impl.construct_roadmap(
                    allowed_time=prm_construction_time
                )

        # self._visualize_roadmap()

        # Add start and goal positions
        subrobot_starts = self._add_subrobot_states_to_prm(start)
        subrobot_goals = self._add_subrobot_states_to_prm(goal)

        # Make the implicit product roadmap
        repeated_prms = []
        for r_idx in range(self.n_subrobots):
            u_r_idx = self.subrobot_to_unique_subrobot_map[r_idx]
            prm = self.prms[u_r_idx]
            repeated_prms.append(prm)

        self.implicit_graph = ImplicitGraph(self.problem, repeated_prms)

        # Solve Discrete Problem
        p = PebbleMotionProblem(
            graph=self.implicit_graph,
            backend="corallab"
        )
        planner = Planner(
            "M_STAR",
            problem=p,
            n_iters=self.n_iters,
            max_time=self.max_time,
            backend="corallab"
        )
        # sol, info =
        return planner.solve(start, goal)
