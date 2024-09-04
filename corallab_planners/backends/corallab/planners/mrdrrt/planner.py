import torch
import numpy as np
import pylab as pl
import time
import random

from torch_robotics.torch_utils.torch_timer import TimerCUDA

from corallab_lib.backends.corallab.implicit_graph import ImplicitGraph
from corallab_lib import MotionPlanningProblem, PebbleMotionProblem

from corallab_planners import Planner

from ..multi_agent_prm_planner import MultiAgentPRMPlanner
from .tree import Tree


class MRdRRT(MultiAgentPRMPlanner):
    """Multi-robot discrete RRT algorithm for coordinated centralized planning.

    Simple implementation of:

    Solovey, Kiril, Oren Salzman, and Dan Halperin.
    "Finding a needle in an exponential haystack: Discrete RRT for exploration
    of implicit roadmaps in multi-robot motion planning." Algorithmic Foundations
    of Robotics XI. Springer International Publishing, 2015. 591-607.
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
        self.visualize = visualize

    def _collision_fn(self, qs, **observation):
        return self.problem.check_collision(qs).squeeze()

    def _oracle(self, q_near, q_rand):
        """Direction oracle, as defined in Oren's paper.
        Given randomly sampled comp config and nearest config on current tree,
        return q_new, a neighbor of qnear on the implicit graph that hasn't been
        explored yet, is , and is closest (by sum of euclidean distances) to qnear.
        """

        q_neighbor = self.implicit_graph.get_closest_composite_neighbor(q_near, q_rand)

        # check collision between q_near and q_neighbor
        path = self.problem.local_motion(q_near, q_neighbor, no_max_dist=True)
        connectable = not self.problem.check_collision(path).any().item()

        if connectable:
            return q_neighbor
        else:
            return None

    def _expand(self, tree, goal):
        """Takes random sample and tries to expand tree in direction of sample.
        """
        if random.random() > 0.3:
            q_rand = self.problem.random_q(n_samples=1).squeeze()
        else:
            q_rand = goal

        q_near, near_id = tree.nearest_neighbors(q_rand, k=1)
        q_near, near_id = q_near.squeeze(), near_id.item()

        q_new = self._oracle(q_near, q_rand)

        if (q_new is not None and q_new not in tree.vertices):
            new_id = tree.add_vertex(q_new)
            tree.add_edge(near_id, new_id)

    def _connect_to_target(self, tree, goal):
        """Check if it's possible to get to goal from closest nodes in current tree.
        Called at the end of each iteration.
        Input: goal composite config
        """
        q_near, near_id = tree.nearest_neighbors(goal, k=1)
        q_near, near_id = q_near.squeeze(), near_id.item()

        path = self.problem.local_motion(q_near, goal, no_max_dist=True)
        connectable = not self.problem.check_collision(path).any().item()

        if connectable:
            return path, near_id
        else:
            return None, None

    def _retrieve_path(self, tree, path_ending, tree_ending_id):
        """Returns final path thru implicit graph to get from start to goal.
        Called when a collision-free path to goal config is found.
        Inputs:
            neighbor_of_goal: (node IDs) node that was successfully connected to goal
            gconfigs: list of configurations of final goal
        """

        path_remainder_l = []

        current_node_id = tree.edges[tree_ending_id]
        while current_node_id != -1:
            current_state = tree.vertices[current_node_id]
            path_remainder_l.insert(0, current_state)
            current_node_id = tree.edges[current_node_id]

        path_remainder = torch.stack(path_remainder_l)
        path = torch.vstack([path_remainder, path_ending])

        return path

    def solve(
            self,
            start,
            goal,
            **kwargs
    ):
        """Main function for MRdRRT. Expands tree to find path from start to goal.
        Inputs: list of start and goal configs for robots.
        """

        breakpoint()

        self.construct_roadmaps()

        # self._visualize_roadmap()
        # breakpoint()

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

        # initialize the tree and root it at the start
        tree = Tree(self.implicit_graph)
        tree.add_vertex(start)

        iteration = -1
        solution = None
        info = {}

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                self._expand(tree, goal)
                path_to_goal, tree_ending_id = self._connect_to_target(tree, goal)

                if path_to_goal is not None:
                    solution = self._retrieve_path(tree, path_to_goal, tree_ending_id)
                    break

        return solution, info

