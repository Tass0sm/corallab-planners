import abc
import pickle

import time
import torch
import numpy as np

from corallab_lib import MotionPlanningProblem, PebbleMotionProblem

from corallab_planners import Planner
from corallab_planners.data_structures import NearestNeighbors

from corallab_lib.backends.corallab.roadmap import Roadmap

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
    """
    "Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces"
          Lydia E. Kavraki, Petr Svestka, Jean-Claude Latombe, and Mark H. Overmars
    """

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            connect_distance: float = 0.2,
            max_random_bounce_steps : int = 5,
            max_retries : int = 20,
            tensor_args: dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        self.problem = problem

        self.connect_distance = connect_distance
        self.max_random_bounce_steps = max_random_bounce_steps
        self.construction_phase_time = 1
        self.max_retries = max_retries
        self.loaded_roadmap = False

        # self.bounce_step = 0.1
        # self.bounce_max_dist = 0.8

        self.nn = NearestNeighbors()
        self.roadmap = Roadmap()
        self.tensor_args = tensor_args

    def _visualize_roadmap(self):
        import torch
        from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer

        tr_s_task = self.problem.problem_impl.task_impl

        tr_visualizer = PlanningVisualizer(
            task=tr_s_task,
            planner=self
        )

        # visualize trajectories in environment
        fig, axs = tr_visualizer.render_robot_trajectories(
            trajs=torch.zeros((1, 64, 2), **self.tensor_args),
            render_planner=True
        )

        fig.show()

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

        for i in range(self.max_retries):
            if not self.loaded_roadmap and i == 0:
                self.construct_roadmap()

            # breakpoint()
            # self._visualize_roadmap()

            self.grow_roadmap_with_samples([q1, q2])

            if q1 not in self.roadmap or q2 not in self.roadmap:
                return None

            p = PebbleMotionProblem(
                graph=self.roadmap,
                backend="corallab"
            )
            planner = Planner("A_STAR", problem=p, backend="corallab")
            sol, info = planner.solve(q1, q2)

            if sol is not None:
                return sol, {}
            else:
                print("Retrying")

        return None, {}

    def render(self, ax, **kwargs):
        self.roadmap.draw(ax)

    #############################################################

    def load_roadmap(self, filename):
        with open(filename, 'rb') as f:
            self.roadmap = pickle.load(f)

        vertices_np = torch.stack([v.q for v in self.roadmap.vertices.values()]).cpu().numpy()
        self.nn.add_points(vertices_np)
        self.loaded_roadmap = True

    def save_roadmap(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.roadmap, f)

    #############################################################

    def construct_roadmap(self, allowed_time : float = 15.0):
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
            free_qs, _ = self.problem.random_coll_free_q(n_samples=100)

            # Necessary for pybullet backend
            free_qs = torch.tensor(free_qs, **self.tensor_args)

            vertices = self.roadmap.add(free_qs)
            vertices_np = torch.stack([v.q for v in vertices]).cpu().numpy()

            self.nn.add_points(vertices_np)
            query_results = self.nn.query(vertices_np)

            for v, knn in zip(vertices, query_results):
                self.add_milestone(v, knn)

    def grow_roadmap_with_samples(self, samples):
        if all(x in self.roadmap for x in samples):
            return

        vertices = self.roadmap.add(samples)
        vertices_np = torch.stack([v.q for v in vertices]).cpu().numpy()
        query_results = self.nn.query(vertices_np)

        for v, knn in zip(vertices, query_results):
            self.add_milestone(v, knn)

        self.nn.add_points(vertices_np)

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
        target_states_np = self.problem.random_q(n_samples=self.max_random_bounce_steps).cpu().numpy()
        bounce_points = []
        prev = start_q;
        last_valid = prev

        # std::pair<State *, double> lastValid;
        for i in range(self.max_random_bounce_steps):
            path = list(self.problem.local_motion(prev, target_states_np[i]))[:-1]

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

    def add_milestone(self, m, knn):
        m.total_connection_attempts = 1;
        m.successful_connection_attempts = 0;

        # # Initialize to its own (dis)connected component.
        # disjointSets_.make_set(m);

        neighbors = [self.roadmap[p] for p in knn]
        for n in neighbors:
            if self._connection_filter(n, m):
                m.total_connection_attempts += 1
                n.total_connection_attempts += 1

                local_motion = self.problem.local_motion(m.q, n.q, no_max_dist=True)
                path = list(local_motion)[:-1]

                # dist = local_motion.diff(dim=0).square().sum(dim=-1).sqrt().sum()
                # print(f"dist: {dist}")
                # if dist > 0.5:
                #     breakpoint()

                all_collision_free = not any(self.collision_fn(q) for q in default_selector(path))
                if all_collision_free:
                    self.roadmap.connect(n, m, path)
                    # uniteComponents(n, m);

        return m;

    def _connection_filter(self, a, b):
        # return self.distance_fn(a.q, b.q) <= self.connect_distance
        return True

    def _connection_strategy(self, v1):
        points = self.nn.query(v1.q)
        vertices = [self.roadmap[p] for p in points]
        return vertices

    # def _connection_strategy(self, v1):
    #     breakpoint()

    #     old_vertices = list(self.roadmap.vertices.values())
    #     neighbors = []

    #     for v2 in old_vertices:
    #         if self.distance_fn(v1.q, v2.q) <= self.connect_distance:
    #             neighbors.append(v2)

    #     return neighbors

    def distance_fn(self, q1, q2):
        return self.problem.distance_q(q1, q2).item()
    # return np.linalg.norm(q2 - q1)

    def collision_fn(self, qs, **observation):
        return self.problem.check_collision(qs).squeeze()
