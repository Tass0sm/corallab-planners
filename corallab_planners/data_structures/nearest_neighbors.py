import numpy as np
from pyflann import FLANN


class NearestNeighbors:
    "FLANN based nearest neighbors datastructure."


    def __init__(self):
        self.flann = FLANN()
        self.params = None

    def add_point(self, x):
        assert x.ndim == 1

        x_set = np.expand_dims(x, 0)
        self.add_points(x_set)

    def add_points(self, xs):
        if self.flann._FLANN__curindex is None:
            self.params = self.flann.build_index(xs)
        else:
            self.flann.add_points(xs)

    def query(self, xs, k=5):
        if xs.ndim == 1:
            xs = np.expand_dims(xs, 0)
        elif xs.ndim != 2:
            raise NotImplementedError()

        idxs, dists = self.flann.nn_index(xs, num_neighbors = k+1)

        # Don't include first neighbor because it's always itself.
        idxs, dists = idxs[:, 1:], dists[:, 1:]
        points = self.flann._FLANN__curindex_data[idxs]

        if points.ndim == 2:
            points = np.expand_dims(points, 1)

        return points
