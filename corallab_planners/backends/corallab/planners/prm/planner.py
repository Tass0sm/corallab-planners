import abc

import time
import torch

from corallab_lib import Task

from corallab_planners.backends.corallab.planners.utils import extend_path
from .utils import Roadmap, SearchNode, to_tuple, a_star

from heapq import heappop, heappush
from collections import deque

from torch.distributions.categorical import Categorical
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


def bisect(sequence):
    sequence = list(sequence)
    indices = set()
    queue = deque([(0, len(sequence)-1)])
    while queue:
        lower, higher = queue.popleft()
        if lower > higher:
            continue
        index = int((lower + higher) / 2.)
        assert index not in indices
        #if is_even(higher - lower):
        yield sequence[index]
        queue.extend([
            (lower, index-1),
            (index+1, higher),
        ])


def bisect_selector(path):
    return bisect(path)


default_selector = bisect_selector # random_selector


class PRM:
    #  "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces"
    #        Lydia E. Kavraki, Petr Svestka, Jean-Claude Latombe, and Mark H. Overmars

    def __init__(
            self,
            task : Task = None,
            connect_distance: float = 0.1,
            max_random_bounce_steps : int = 5,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        self.task = task

        self.connect_distance = connect_distance
        self.max_random_bounce_steps = max_random_bounce_steps
        self.construction_phase_time = 1

        self.roadmap = Roadmap()
        self.tensor_args = tensor_args

    def solve(
            self,
            start,
            goal,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """

        # Add start and goal states to roadmap. If either are in
        # collision, no path exists.
        q1 = start
        q2 = goal

        self.grow_roadmap_with_samples([q1, q2])

        return a_star(self, q1, q2)

    def render(self, ax, **kwargs):
        self.roadmap.draw(ax)

    #############################################################

    def construct_roadmap(self, allowed_time : float = 10.0):
        grow = True;

        timeout = time.time() + allowed_time
        while time.time() < timeout:
            if grow:
                self.grow_roadmap(2 * self.construction_phase_time)
            else:
                self.expand_roadmap(self.construction_phase_time)

            grow = not grow

        return self.roadmap

    def grow_roadmap(self, allowed_time):
        timeout = time.time() + allowed_time
        while time.time() < timeout:
            free_qs, _ = self.task.random_coll_free_q(n_samples=100)
            vertices = self.roadmap.add(free_qs)

            for v in vertices:
                self.add_milestone(v)

    def grow_roadmap_with_samples(self, samples):
        vertices = self.roadmap.add(samples)
        for v in vertices:
            self.add_milestone(v)

    def expand_roadmap(self, allowed_time):
        vertices = list(self.roadmap.vertices.values())
        weights = list(map(lambda v: (v.total_connection_attempts - v.successful_connection_attempts) / v.total_connection_attempts, vertices))
        weights = torch.tensor(weights)
        probs = weights / weights.sum()
        dist = Categorical(probs=probs)

        timeout = time.time() + allowed_time
        while time.time() < timeout:
            i = dist.sample()
            v = vertices[i]

            bounce_points = self._random_bounce_motion(v.q);
            bounce_vertices = self.roadmap.add(bounce_points)

            if len(bounce_vertices) > 0:
                self.add_milestone(bounce_vertices[-1])

                for v1, v2 in zip([v, *bounce_vertices], [bounce_vertices]):
                    self.roadmap.connect(v1, v2)

    def _random_bounce_motion(self, start_q):
        target_states = self.task.random_q(n_samples=self.max_random_bounce_steps)
        bounce_points = []
        prev = start_q;
        last_valid = prev

        # std::pair<State *, double> lastValid;
        for i in range(self.max_random_bounce_steps):
            path = list(self._extend_fn(prev, target_states[i]))[:-1]

            for q in path:
                if self.collision_fn(q):
                    last_valid = q
                else:
                    break

            last_valid_distance = self.distance_fn(prev, last_valid)

            if last_valid_distance > 0.001:
                bounce_points.append(last_valid)
                prev = last_valid

        return bounce_points

    def add_milestone(self, m):
        m.total_connection_attempts = 1;
        m.successful_connection_attempts = 0;

        # # Initialize to its own (dis)connected component.
        # disjointSets_.make_set(m);

        # Which milestones will we attempt to connect to?
        neighbors = self._connection_strategy(m);

        for n in neighbors:
            if self._connection_filter(n, m):
                m.total_connection_attempts += 1
                n.total_connection_attempts += 1

                # TODO Make this a little more like OMPL
                path = list(self._extend_fn(n.q, m.q))[:-1]
                all_collision_free = not any(self.collision_fn(q) for q in default_selector(path))
                if all_collision_free:
                    # const base::Cost weight = opt_->motionCost(stateProperty_[n], stateProperty_[m]);
                    # const Graph::edge_property_type properties(weight);
                    # boost::add_edge(n, m, properties, g_);
                    self.roadmap.connect(n, m, path)
                    # uniteComponents(n, m);

        # nn_->add(m);
        return m;

    def _connection_filter(self, a, b):
        return True

    def _extend_fn(self, q1, q2, max_step=0.03, max_dist=0.1):
        return extend_path(self.distance_fn, q1, q2, max_step, max_dist, tensor_args=self.tensor_args)

    def _connection_strategy(self, v1):
        old_vertices = list(self.roadmap.vertices.values())
        neighbors = []

        for v2 in old_vertices:
            if self.distance_fn(v1.q, v2.q) <= self.connect_distance:
                neighbors.append(v2)

        return neighbors


    # def grow_roadmap(self, samples):
    #     old_vertices = list(self.roadmap.vertices.values())
    #     new_vertices = self.roadmap.add(samples)

    #     for i, v1 in enumerate(new_vertices):
    #         for v2 in new_vertices[i + 1:] + old_vertices:
    #             if self.distance_fn(v1.q, v2.q) <= self.connect_distance:
    #                 path = list(self.extend_fn(v1.q, v2.q))[:-1]
    #                 all_collision_free = not any(self.collision_fn(q) for q in default_selector(path))
    #                 if all_collision_free:
    #                     self.roadmap.connect(v1, v2, path)

    #     return new_vertices

    def distance_fn(self, q1, q2):
        # TODO: cleanup
        return self.task.distance_q(q1, q2).item()

    def collision_fn(self, qs, **observation):
        # TODO: clean
        return self.task.compute_collision(qs).squeeze()
