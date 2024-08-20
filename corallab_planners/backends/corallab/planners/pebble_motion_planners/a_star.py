import torch
import numpy as np
from heapq import heappop, heappush

from corallab_lib import PebbleMotionProblem
from .search_node import SearchNode


class A_STAR:

    def __init__(
            self,
            problem : PebbleMotionProblem,
    ):
        assert problem.n_pebbles == 1, "A_STAR only works with one pebble"

        self.problem = problem
        self.graph = problem.graph

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
        if start not in self.graph or goal not in self.graph:
            return None

        start_v, goal_v = self.graph[start], self.graph[goal]
        heuristic = lambda v: self.problem.distance_fn(v.q, goal_v.q)  # lambda v: 0

        queue = [(heuristic(start_v), start_v)]
        nodes, processed = {start_v: SearchNode(0, None)}, set()
        solution_l = None

        # def retrace(v):
        #     print(f"Retracing from {v}")
        #     if nodes[v].parent is None:
        #         return [v.q]

        #     v.edges[nodes[v].parent].in_shortest_path = True
        #     # return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)
        #     return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)

        while len(queue) != 0:
            _, cv = heappop(queue)
            if cv in processed:
                continue
            processed.add(cv)

            if cv == goal_v:
                solution_l = self._retrace(cv, nodes)
                break

            for nv in cv.edges:
                cost = nodes[cv].cost + self.problem.distance_fn(cv.q, nv.q)
                if (nv not in nodes) or (cost < nodes[nv].cost):
                    nodes[nv] = SearchNode(cost, cv)
                    heappush(queue, (cost + heuristic(nv), nv))

        if solution_l is not None:
            solution = torch.stack(solution_l).unsqueeze(0)
            return solution, {}
        else:
            return None, {}
