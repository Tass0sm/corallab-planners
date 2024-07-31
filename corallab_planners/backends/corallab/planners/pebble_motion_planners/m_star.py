import torch
import itertools
import numpy as np
from heapq import heappop, heappush

try:
    from collections import namedtuple
except ImportError:
    from collections.abc import namedtuple

from torch_robotics.torch_utils.torch_timer import TimerCUDA

from corallab_lib import PebbleMotionProblem
from corallab_lib.backends.corallab.roadmap import Roadmap
from corallab_lib.backends.corallab.implicit_graph import ImplicitGraph

SearchNode = namedtuple('SearchNode', ['cost', 'parent'])


class M_STAR:

    def __init__(
            self,
            problem : PebbleMotionProblem,
            n_iters: int = 30000,
            max_time: float = 60.,
            **kwargs
    ):
        assert isinstance(problem.graph, ImplicitGraph), "M_STAR needs a multiagent roadmap"

        self.n_iters = n_iters
        self.max_time = max_time

        self.problem = problem
        self.graph = problem.graph
        self.roadmaps = problem.graph.roadmaps

    def _single_goal_shortest_paths(self, roadmap, goal_v):
        heuristic = lambda v: self.problem.distance_fn(v.q, goal_v.q)

        queue = [(heuristic(goal_v), goal_v)]
        nodes, processed = {goal_v: SearchNode(0, goal_v)}, set()
        solution_l = None

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)

            for nv in cv.edges:
                cost = nodes[cv].cost + self.problem.distance_fn(cv.q, nv.q)
                if (nv not in nodes) or (cost < nodes[nv].cost):
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))

        return nodes

    def _limited_neighbors(self, state_v):
        """
        Assumes state is collision free
        """

        collision_set = self.collision_set_dict.get(state_v, {})
        subrobot_states = self.graph._separate_subrobot_states(state_v.q)

        neighbors_l = []

        for i, state_i in enumerate(subrobot_states):
            roadmap = self.roadmaps[i]
            vertex_i = roadmap[state_i]

            if i in collision_set:
                # Includes self, which is appropriate because sometimes the best
                # neighbor state requires not moving robot_i
                neighbors_i = list(vertex_i.edges)
            else:
                policy_i = self.policy_dict[i]
                neighbors_i = [policy_i[vertex_i].parent]

            neighbors_l.append(neighbors_i)

        # Return all possible combinations of neighbors
        neighbors = list(itertools.product(*neighbors_l))
        neighbor_states = list(map(lambda p: torch.cat([v.q for v in p]), neighbors))
        neighbor_vertices = self.explicit_roadmap.add_or_get(neighbor_states)

        return neighbor_vertices

    def _backprop(self, cv, nv, nodes, heuristic):
        c_k = self.collision_set_dict.get(cv, {})
        c_l = self.collision_set_dict.get(nv, {})

        if not c_l.issubset(c_k):
            self.collision_set_dict[cv] = c_k.union(c_l)

            if cv not in [vert for (cost, vert) in self.open_set]:
                parent = nodes[cv].parent

                if parent is not None:
                    cost = nodes[parent].cost + self.problem.distance_fn(parent.q, cv.q)
                else:
                    cost = 0

                heappush(self.open_set, (cost + heuristic(cv), cv))

            for mv in self.backprop_set_dict[cv]:
                self._backprop(mv, cv, nodes, heuristic)

    def _retrace(self, v, nodes):
        if v is None:
            return []

        if nodes[v].parent is v:
            return [v.q]

        parent = nodes[v].parent
        return self._retrace(parent, nodes) + [v.q]

    def solve(
            self,
            start,
            goal,
            **kwargs
    ):
        """
        Plan motion in implicit graph.
        """

        if start not in self.graph or goal not in self.graph:
            return None, {}

        subrobot_start_vs = self.graph[start]
        subrobot_goal_vs = self.graph[goal]

        # initialize policy table
        self.policy_dict = {}
        for i, (roadmap, goal_v_i) in enumerate(zip(self.roadmaps, subrobot_goal_vs)):
            self.policy_dict[i] = self._single_goal_shortest_paths(roadmap, goal_v_i)

        # Make the explicit roadmap
        self.explicit_roadmap = Roadmap()
        start_v, goal_v = self.explicit_roadmap.add([start, goal])

        # TODO: It might be more idiomatic to implement these as instance
        # variables of a vertex object, but that might complicate reusing the
        # roadmap class
        self.collision_set_dict = { start_v: set() }
        self.backprop_set_dict = { start_v: set() }

        heuristic = lambda v: self.problem.distance_fn(v.q, goal_v.q)
        self.open_set = [(heuristic(start_v), start_v)]
        nodes, processed = {start_v: SearchNode(0, None)}, set()

        iteration = -1
        solution_l = None
        info = {}

        print("Searching the composite PRM")

        with TimerCUDA() as t:
            # while (t.elapsed < self.max_time) and (iteration < self.n_iters) and len(self.open_set) != 0:
            while (iteration < self.n_iters) and len(self.open_set) != 0:
                iteration += 1

                _, cv = heappop(self.open_set)

                if cv in processed:
                    # skip vertices that have already been processed
                    print("Redundant!")
                    # continue
                else:
                    processed.add(cv)

                # TODO: Check if margin=0 is unnecessary
                # cv.q not in self.graph:
                if not self.graph.__contains__(cv.q, margin=0):
                    breakpoint()
                    print(f"Discarding {cv} due to collision")
                    # skip vertices that are in collision
                    continue

                if (cv.q == goal).all():
                    solution_l = self._retrace(cv, nodes)
                    break

                for nv in self._limited_neighbors(cv):
                    cost = nodes[cv].cost + self.problem.distance_fn(cv.q, nv.q)

                    if nv not in self.backprop_set_dict:
                        self.backprop_set_dict[nv] = set()

                    if nv not in self.collision_set_dict:
                        self.collision_set_dict[nv] = set()

                    self.backprop_set_dict[nv].add(cv)

                    # TODO: Really need to use margin=0?????
                    if self.problem.check_local_motion(cv.q, nv.q, margin=0):
                        if (nv not in nodes) or (cost < nodes[nv].cost):
                            # print(f"Relaxing {nv} with cost {cost}")
                            nodes[nv] = SearchNode(cost, cv)
                            heappush(self.open_set, (cost + heuristic(nv), nv))
                    else:
                        print(f"found a self collision between {cv} and {nv}")
                        self_collision_robots = [0, 1] # info["self_collision_robots"][:, 1:].flatten().tolist()
                        self.collision_set_dict[nv] = self.collision_set_dict[nv].union(self_collision_robots)
                        self._backprop(cv, nv, nodes, heuristic)

        if solution_l is None:
            return None, {}
        else:
            solution = torch.stack(solution_l).unsqueeze(0)
            return solution, {}
