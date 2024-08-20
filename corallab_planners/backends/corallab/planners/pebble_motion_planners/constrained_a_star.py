import torch
import numpy as np
from heapq import heappop, heappush
from copy import copy

try:
    from collections import namedtuple
except ImportError:
    from collections.abc import namedtuple

from corallab_lib import PebbleMotionProblem


SearchNode = namedtuple('SearchNode', ['cost', 'parent'])


class CONSTRAINED_A_STAR:
    """Running Constrained A Star on a graph G is like running A Star over a
    transformed graph G' with infinite copies of every node for every timestep
    going to infiity. However the algorithm should terminate when reaching any
    of the goal node copies, as long as no future copy of the goal node conflicts
    with something in the constraint set.

    Also, extra care should be taken when checking if nodes have already been
    processed.

    Nodes x and y in G' are only connected by a directed edge if they are
    connected in the original graph and y's timestep is equal to x's timestep +
    1.

    If x and y correspond to the same node in G but in different timesteps they
    are connected with a directed edge with an associated arbitrary weight
    cost. The weight cost is 2 by default. If the original graph's edge weights
    are significantly higher than the waiting cost, the algorithm will slow down
    because it will relax over waiting edges many times.

    If the waiting edge weight is zero, the algorithm will forever stay in
    place.

    TODO: Investigate and explain choices of heuristic

    """

    def __init__(
            self,
            problem : PebbleMotionProblem,
            waiting_edge_weight : float = 2.0,
    ):
        assert problem.n_pebbles == 1, "CONSTRAINED_A_STAR only works with one pebble"

        self.problem = problem
        self.graph = problem.graph
        self.waiting_edge_weight = waiting_edge_weight

    def _retrace(self, v, nodes):
        if v is None:
            return []

        if nodes[v].parent is v:
            return [v.q]

        parent = nodes[v].parent
        return self._retrace(parent, nodes) + [v.q]

    def _is_constrained(self, v1, v2, constraints):
        """TODO: This should make use of the problem's check_local_motion method"""
        if constraints is None:
            return False
        else:
            for c in constraints:
                if v.time in c.time:
                    idx = (c.time == v.time).argwhere().squeeze()
                    other_state = c.traj[idx]

                    if main_i == 0 and c.i == 1:
                        main_state = torch.tensor(v.q, **task.tensor_args)
                        joint_state = torch.cat([main_state, other_state])
                    elif main_i == 1 and c.i == 0:
                        main_state = torch.tensor(v.q, **task.tensor_args)
                        joint_state = torch.cat([other_state, main_state])
                    else:
                        raise NotImplementedError

                    in_coll = task.check_collision(joint_state)
                    if in_coll:
                        return True

            return False

    def _constrained_in_future(self, v, constraints):
        """TODO"""
        return False

    def solve(
            self,
            start,
            goal,
            constraints=None,
            **kwargs
    ):
        if start not in self.graph or goal not in self.graph:
            return None

        start_v, goal_v = self.graph[start], self.graph[goal]
        start_v = copy(start_v)
        start_v.time = 0

        heuristic = lambda v: self.problem.distance_fn(v.q, goal_v.q)  # lambda v: 0

        queue = [(heuristic(start_v), start_v)]
        nodes, processed = {start_v: SearchNode(0, None)}, set()
        solution_l = None

        while len(queue) != 0:
            _, cv = heappop(queue)

            if cv in processed:
                continue

            processed.add(cv)

            if (cv.q == goal_v.q).all():
                if not self._constrained_in_future(cv, constraints):
                    solution_l = self._retrace(cv, nodes)
                    break

            any_constrained = False

            for nv in cv.edges:
                nv = copy(nv)
                nv.time = cv.time + 1
                cost = nodes[cv].cost + self.problem.distance_fn(cv.q, nv.q)

                constrained = self._is_constrained(cv, nv, constraints)

                if ((nv not in nodes) or (cost < nodes[nv].cost)) and not constrained:
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))
                elif constrained:
                    any_constrained = True

            # Also relax over waiting edge (self loop) to allow waiting in place
            # Untested optimization: only wait a constrained state was found.
            if any_constrained:
                nv = copy(cv)
                nv.time = cv.time + 1
                cost = nodes[cv].cost + self.waiting_edge_weight
                constrained = self._is_constrained(cv, nv, constraints)
                if not constrained:
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))


        if solution_l is not None:
            solution = torch.stack(solution_l).unsqueeze(0)
            return solution, {}
        else:
            return None, {}
