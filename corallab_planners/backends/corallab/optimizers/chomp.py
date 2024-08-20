import math
import torch

from corallab_lib import MotionPlanningProblem

from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import (
    batched_weighted_dot_prod,
    DEFAULT_TENSOR_ARGS,
    tensor_linspace_v1,
    to_torch
)


class CHOMP:

    def __init__(
            self,
            problem : MotionPlanningProblem = None,
            n_support_points : int = 64,
            dt : float = 0.4,
            opt_iters : int = 1,
            step_size : float = 0.1,
            grad_clip : float = .01,
            weight_prior_cost : float = 0.1,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        # super(CHOMP, self).__init__(name='CHOMP',
        #                             n_dof=n_dof,
        #                             n_support_points=n_support_points,
        #                             num_particles_per_goal=num_particles_per_goal,
        #                             opt_iters=opt_iters,
        #                             dt=dt,
        #                             start_state=start_state,
        #                             cost=cost,
        #                             initial_particle_means=initial_particle_means,
        #                             multi_goal_states=multi_goal_states,
        #                             sigma_start_init=sigma_start_init,
        #                             sigma_goal_init=sigma_goal_init,
        #                             sigma_gp_init=sigma_gp_init,
        #                             pos_only=pos_only,
        #                             **kwargs)

        self.opt_iters = opt_iters
        self.dt = dt
        self.n_support_points = n_support_points
        self.tensor_args = tensor_args

        # CHOMP params
        self.lr = step_size
        self.grad_clip = grad_clip

        self._particle_means = None
        # Precision matrix, shape: [ctrl_dim, n_support_points, n_support_points]
        self.Sigma_inv = self._get_R_mat(dt=self.dt, n_support_points=self.n_support_points, tensor_args=self.tensor_args)
        self.Sigma = torch.inverse(self.Sigma_inv)

        # Weight prior
        self.weight_prior_cost = weight_prior_cost

    @property
    def name(self):
        return "chomp"

    @classmethod
    def _get_R_mat(
            cls,
            dt=0.01,
            n_support_points=64,
            tensor_args={},
            **kwargs
    ):
        """
        CHOMP time-correlated Precision matrix.
        Backward finite difference velocity.
        """
        lower_diag = -torch.diag(torch.ones(n_support_points - 1), diagonal=-1)
        diag = 1 * torch.eye(n_support_points)
        K_mat = diag + lower_diag
        K_mat = torch.cat((K_mat, torch.zeros(1, n_support_points)), dim=0)
        K_mat[-1, -1] = -1.
        K_mat = K_mat * 1. / dt ** 2
        R_mat = K_mat.t() @ K_mat

        return R_mat.to(**tensor_args)

    def _interpolate_states(self, states):
        request_count = self.n_support_points
        states_l = [s for s in states]

        if request_count < len(states) or len(states) < 2:
            raise NotImplementedError

        count = request_count;

        # the remaining length of the path we need to add states along
        remaining_length = states.diff(dim=0).square().sum(dim=-1).sqrt().sum()

        # the new array of states this path will have
        new_states = []
        n1 = len(states_l) - 1;

        for i in range(n1):
            s1 = states_l[i];
            s2 = states_l[i + 1];

            new_states.append(s1)

            # the maximum number of states that can be added on the current motion (without its endpoints)
            # such that we can at least fit the remaining states
            max_n_states = count + i - len(states_l)

            if max_n_states > 0:
                # compute an approximate number of states the following segment needs to contain; this includes endpoints
                segment_length = torch.linalg.norm(s2 - s1)
                ns = max_n_states + 2 if i + 1 == n1 else math.floor(0.5 + count * segment_length / remaining_length) + 1;

                if ns > 2:
                    if (ns - 2 > max_n_states):
                        ns = max_n_states + 2

                    segment_states = tensor_linspace_v1(
                        s1, s2, steps=ns
                    ).T
                    segment_states = [s for s in segment_states[1:-1]]
                    new_states.extend(segment_states)
                    ns -= 2
                else:
                    ns = 0

                # update what remains to be done
                count -= (ns + 1)
                remaining_length -= segment_length
            else:
                count -= 1

        # add the last state
        new_states.append(states_l[n1])
        states = torch.stack(new_states)

        return states

    def reset(
            self,
            initial_particle_means=None,
    ):
        # Straight line position-trajectory from start to goal with const vel
        if initial_particle_means is not None:
            self._particle_means = initial_particle_means.clone()
        else:
            self._particle_means = self.get_random_trajs()

    def optimize(
            self,
            guess=None,
            objective=None,
            opt_iters=None,
            **observation
    ):

        if opt_iters is None:
            opt_iters = self.opt_iters

        guess = to_torch(guess, **self.tensor_args)

        # if guess.ndim == 2:
        #     guess = guess.unsqueeze(0)

        # if guess.shape[1] <= self.n_support_points:
        #     guess = self._interpolate_states(guess)



        self.reset(initial_particle_means=guess)

        solution_l = []
        grad_l = []

        for opt_step in range(opt_iters):
            self._particle_means.requires_grad_(True)
            obj_value = self._eval(self._particle_means, objective)

            # Get grad
            obj_value.sum().backward(retain_graph=True)
            # For stabilizing and preventing high gradients
            self._particle_means.grad.data.clamp_(-self.grad_clip, self.grad_clip)
            # zeroing grad at start and goal
            self._particle_means.grad.data[..., 0, :] = 0.
            self._particle_means.grad.data[..., -1, :] = 0.

            # Update trajectory
            grad_l.append((-self.lr * self._particle_means.grad.data).clone().detach())
            solution_l.append(self._particle_means.clone().detach())

            self._particle_means.data.add_(-self.lr * self._particle_means.grad.data)
            self._particle_means.grad.detach_()
            self._particle_means.grad.zero_()

        grads_iters = torch.stack(grad_l)
        solution_iters = torch.stack(solution_l)

        # get mean trajectory
        curr_traj = self._particle_means.detach()

        return curr_traj, { "grads_iters": grads_iters, "solution_iters": solution_iters }

    def _eval(self, x, objective):
        """
        Evaluate costs.
        """
        if x.ndim == 2:
            x = x.unsqueeze(0)
        elif x.ndim == 4:
            x = x.squeeze(0)

        # Evaluate (collision, ...) costs
        if objective is not None:
            costs = objective(x)
        else:
            costs = 0

        # Add smoothness term (prior)
        R_mat = self.Sigma_inv

        smooth_cost = batched_weighted_dot_prod(x, R_mat, x)
        smooth_cost = smooth_cost.sum()
        costs += self.weight_prior_cost * smooth_cost

        return costs
