# From: https://github.com/caelan/motion-planners/master/motion_planners/prm.py

import torch
import numpy as np
from copy import copy

try:
    from collections import Mapping, namedtuple
except ImportError:
    from collections.abc import Mapping, namedtuple

from heapq import heappop, heappush
import matplotlib.pyplot as plt
from matplotlib import patches


class Vertex(object):

    def __init__(self, q):
        self.q = q
        self.time = None
        self.edges = {}
        self._handle = None
        self.total_connection_attempts = 0
        self.successful_connection_attempts = 0

    def __lt__(self, other):
        # return (self.q.norm() < other.q.norm()).item()
        return (np.linalg.norm(self.q) < np.linalg.norm(other.q))

    def __le__(self, other):
        # return (self.q.norm() <= other.q.norm()).item()
        return (np.linalg.norm(self.q) <= np.linalg.norm(other.q))

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return (self.q == other.q).all() and (self.time == other.time)

    def clear(self):
        self._handle = None

    def draw(self, ax, color="red"):
        assert len(self.q) == 2
        p = self.q # .cpu().numpy()
        x, y = p[0], p[1]
        circle = plt.Circle((x, y), 0.005, color="black", linewidth=0, alpha=1)
        ax.add_patch(circle)

    def __str__(self):
        return 'Vertex(' + str(self.q) + ')'

    __repr__ = __str__


class Edge(object):

    def __init__(self, v1, v2, path):
        self.v1, self.v2 = v1, v2
        self.v1.edges[v2], self.v2.edges[v1] = self, self
        self._path = path
        #self._handle = None
        self._handles = []

        self.in_shortest_path = False

    def end(self, start):
        if self.v1 == start:
            return self.v2
        if self.v2 == start:
            return self.v1
        assert False

    def path(self, start):
        if self._path is None:
            return [self.end(start).q]
        if self.v1 == start:
            return self._path + [self.v2.q]
        if self.v2 == start:
            return self._path[::-1] + [self.v1.q]
        assert False

    def configs(self):
        if self._path is None:
            return []
        return [self.v1.q] + self._path + [self.v2.q]

    def clear(self):
        #self._handle = None
        self._handles = []

    def draw(self, ax, color="red"):
        if self._path is None:
            return

        assert len(self.v1.q) == 2 and len(self.v2.q) == 2
        p1 = self.v1.q # .cpu().numpy()
        p2 = self.v2.q # .cpu().numpy()
        dx, dy = p2 - p1

        color = "orange" if self.in_shortest_path else "black"
        width = 0.01 if self.in_shortest_path else 0.00001
        zorder = 100 if self.in_shortest_path else 0

        line = patches.FancyArrow(p1[0], p1[1], dx, dy, width=width, head_length=0,
                                  color=color, zorder=zorder)
        ax.add_patch(line)

    def __str__(self):
        return 'Edge(' + str(self.v1.q) + ' - ' + str(self.v2.q) + ')'

    __repr__ = __str__


SearchNode = namedtuple('SearchNode', ['cost', 'parent'])


def to_tuple(q):
    return tuple([x.item() for x in q])


class Roadmap(Mapping, object):

    def __init__(self, samples=[]):
        self.vertices = {}
        self.edges = []
        self.add(samples)

    def __getitem__(self, q):
        if isinstance(q, torch.Tensor):
            q = to_tuple(q)
        elif isinstance(q, np.ndarray):
            q = to_tuple(q)

        return self.vertices[q]

    def __contains__(self, q):
        if isinstance(q, torch.Tensor):
            q = to_tuple(q)
        elif isinstance(q, np.ndarray):
            q = to_tuple(q)

        return q in self.vertices

    def __len__(self):
        return len(self.vertices)

    def __iter__(self):
        return iter(self.vertices)

    def add(self, samples):
        new_vertices = []
        for q in samples:
            key = to_tuple(q)
            if key not in self:
                self.vertices[key] = Vertex(q)
                new_vertices.append(self[key])

        return new_vertices

    def add_or_get(self, samples):
        vertices = []
        for q in samples:
            key = to_tuple(q)
            if key not in self:
                self.vertices[key] = Vertex(q)
                vertices.append(self[key])
            else:
                vertices.append(self[key])

        return vertices

    def connect(self, v1, v2, path=None):
        if v1 not in v2.edges:  # TODO - what about parallel edges?
            edge = Edge(v1, v2, path)
            self.edges.append(edge)
            return edge
        return None

    def clear(self):
        for v in self.vertices.values():
            v.clear()
        for e in self.edges:
            e.clear()

    def draw(self, ax):
        for v in self.vertices.values():
            v.draw(ax)
        for e in self.edges:
            e.draw(ax)

    @staticmethod
    def merge(*roadmaps):
        new_roadmap = Roadmap()
        new_roadmap.vertices = merge_dicts(
            *[roadmap.vertices for roadmap in roadmaps])
        new_roadmap.edges = list(
            flatten(roadmap.edges for roadmap in roadmaps))
        return new_roadmap


def a_star(prm, q1, q2):
    if q1 not in prm.roadmap or q2 not in prm.roadmap:
        return None

    # A*
    start, goal = prm.roadmap[q1], prm.roadmap[q2]
    heuristic = lambda v: prm.distance_fn(v.q, goal.q)  # lambda v: 0

    queue = [(heuristic(start), start)]
    nodes, processed = {start: SearchNode(0, None)}, set()
    solution_l = None

    def retrace(v):
        if nodes[v].parent is None:
            return [v.q]

        v.edges[nodes[v].parent].in_shortest_path = True
        return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)

    while len(queue) != 0:
        _, cv = heappop(queue)
        if cv in processed:
            continue
        processed.add(cv)

        if cv == goal:
            solution_l = retrace(cv)
            break

        for nv in cv.edges:
            cost = nodes[cv].cost + prm.distance_fn(cv.q, nv.q)
            if (nv not in nodes) or (cost < nodes[nv].cost):
                nodes[nv] = SearchNode(cost, cv)
                heappush(queue, (cost + heuristic(nv), nv))

    return solution_l


TimedSearchNode = namedtuple('TimedSearchNode', ['cost', 'time', 'parent'])


def constrained(task, main_i, v, constraints):
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

# space time a_star with constraints
def constrained_a_star(task, prm, main_i, q1, q2, goal_time, constraints):
    if q1 not in prm.roadmap or q2 not in prm.roadmap:
        return None

    # A*
    start, goal = prm.roadmap[q1], prm.roadmap[q2]
    start = copy(start)
    start.time = 0

    goal.time = goal_time
    heuristic = lambda v: prm.distance_fn(v.q, goal.q)  # lambda v: 0

    queue = [(heuristic(start), start)]
    nodes, processed = {start: SearchNode(0, None)}, set()
    solution_l = None
    prev = None

    def retrace(v):
        if nodes[v].parent is None:
            return [v.q]

        v.edges[nodes[v].parent].in_shortest_path = True
        return retrace(nodes[v].parent) + v.edges[nodes[v].parent].path(nodes[v].parent)

    while len(queue) != 0:
        _, cv = heappop(queue)

        if cv in processed:
            continue

        processed.add(cv)

        if cv == goal:
            solution_l = retrace(cv)
            break

        for nv in cv.edges:
            nv = copy(nv)
            nv.time = cv.time + 1
            cost = nodes[cv].cost + prm.distance_fn(cv.q, nv.q)
            if ((nv not in nodes) or (cost <= nodes[nv].cost)) and not constrained(task, main_i, nv, constraints):
                nodes[nv] = SearchNode(cost, cv)
                heappush(queue, (cost + heuristic(nv), nv))

        # Also relax over self loop to allow waiting in place
        nv = copy(cv)
        nv.time = cv.time + 1
        cost = nodes[cv].cost + prm.distance_fn(cv.q, nv.q) # should be 0
        if not constrained(task, main_i, nv, constraints):
            nodes[nv] = SearchNode(cost, cv)
            heappush(queue, (cost + heuristic(nv), nv))

        prev = cv


    return solution_l
