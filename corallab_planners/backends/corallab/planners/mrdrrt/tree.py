import torch
import operator
from collections import defaultdict
import itertools


class Tree:
    """Tree structure, meant for use with implicit graph in MRdRRT.
    In this class, 'configuration' refers to a list of PRM node IDs, which
    correspond to positions of the robots.
    Adjacency list representation.
    """

    def __init__(self, implicit_graph):
        self.vertices = torch.empty(0, implicit_graph.get_n_dof(), **implicit_graph.task.tensor_args)
        self.edges = { 0: -1 }
        self.implicit_graph = implicit_graph

    def add_vertex(self, config):
        """Add vertex to tree."""
        vid = len(self.vertices)
        self.vertices = torch.vstack((self.vertices, config))
        return vid

    def add_edge(self, sid, eid):
        """Add edge to tree.
        Each node points to its parent (where it came from), which helps for
        reconstructing path at end.
        """
        self.edges[eid] = sid

    def nearest_neighbors(self, config, k=1):
        """Given composite configuration, find K closest ones in current tree.
        """
        config = config.expand(self.vertices.shape)
        dist = self.vertices.add(-config).square().sum(dim=-1).sqrt()
        knn_indices = dist.topk(k, largest=False, sorted=False)[1]
        return self.vertices[knn_indices], knn_indices
