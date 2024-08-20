import abc

import torch


class RRTBase:

    def __init__(
            self,
            problem=None,
            n_iters: int = None,
            step_size: float = 0.1,
            n_radius: float = 1.,
            max_time: float = 60.,
            tensor_args: dict = None,
            n_pre_samples=10000,
            pre_samples=None,
            **kwargs
    ):
        self.tensor_args = tensor_args

        self.problem = problem
        self.n_iters = n_iters

        # RRT params
        self.step_size = step_size
        self.n_radius = n_radius
        self.max_time = max_time

        self.n_pre_samples = n_pre_samples
        self.pre_samples = pre_samples
        self.last_sample_idx = None
        self.n_samples_refill = self.n_pre_samples

        self.reset()

    def reset(self):
        # Create pre collision-free samples
        n_uniform_samples = self.n_pre_samples - (self.pre_samples.shape[0] if self.pre_samples is not None else 0)
        uniform_samples = self.create_uniform_samples(n_uniform_samples)
        if self.pre_samples is not None:
            self.pre_samples = torch.cat((self.pre_samples, uniform_samples), dim=0)
        else:
            self.pre_samples = uniform_samples

    def create_uniform_samples(self, n_samples, max_samples=1000, **observation):
        qs, base_poses = self.problem.random_coll_free_q(n_samples, max_samples)
        return qs

    def remove_last_pre_sample(self):
        # https://discuss.pytorch.org/t/how-to-remove-an-element-from-a-1-d-tensor-by-index/23109/3
        if len(self.pre_samples) > 0:
            i = self.last_sample_idx
            self.pre_samples = torch.cat([self.pre_samples[:i], self.pre_samples[i+1:]])

    def optimize(
            self,
            opt_iters=None,
            **observation
    ):
        """
        Optimize for best trajectory at current state
        """
        return self._run_optimization(opt_iters, **observation)

    @abc.abstractmethod
    def _run_optimization(self, opt_iters, **observation):
        raise NotImplementedError

    def random_collision_free(self, **observation):
        """
        Returns: random positions in environments space not in collision
        """
        refill_samples_buffer = observation.get('refill_samples_buffer', False)
        if len(self.pre_samples) > 0:
            qs = self.get_pre_sample()
        elif refill_samples_buffer:
            self.pre_samples = self.create_uniform_samples(self.n_samples_refill, **observation)
            qs = self.get_pre_sample()
        else:
            qs = self.create_uniform_samples(1, **observation)

        return qs

    def get_pre_sample(self):
        idx = torch.randperm(len(self.pre_samples))[0]
        qs = self.pre_samples[idx]
        self.last_sample_idx = idx
        return qs

    def collision_fn(self, qs, time=0, **observation):
        return self.problem.check_collision(time, qs).squeeze()

    def sample_fn(self, without_collision=True, **observation):
        if without_collision:
            return self.random_collision_free(**observation)
        else:
            return self.task.random_q()

    def distance_fn(self, q1, q2):
        return self.problem.distance_q(q1, q2)

    def get_nearest_node(self, nodes, nodes_torch, target):
        distances = self.distance_fn(nodes_torch, target)
        min_idx = torch.argmin(distances)
        nearest = nodes[min_idx]
        return nearest
