import torch
import numpy as np

from corallab_lib import MotionPlanningProblem

from . import optimizers
from ..optimizer_interface import OptimizerInterface

from torch_robotics.torch_utils.torch_utils import to_torch, tensor_linspace_v1
from torch_robotics.trajectory.utils import smoothen_trajectory


class CorallabOptimizer(OptimizerInterface):

    def __init__(
            self,
            optimizer_name : str,
            problem : MotionPlanningProblem = None,
            add_noise_before_optimization : bool = False,
            noise_std : float = 0.1,
            **kwargs
    ):
        self.tensor_args = problem.tensor_args

        self.add_noise_before_optimization = add_noise_before_optimization
        self.noise_std = noise_std

        OptimizerClass = getattr(optimizers, optimizer_name)
        self.optimizer_impl = OptimizerClass(
            problem=problem,
            **kwargs
        )

    @property
    def name(self):
        return f"corallab_{self.optimizer_impl.name}"

    def optimize(self, guess=None, **kwargs):
        if isinstance(guess, list):
            initial_traj_pos_vel = self._handle_guess_list(guess)
        elif isinstance(guess, torch.Tensor):
            guess_l = [x for x in guess]
            initial_traj_pos_vel = self._handle_guess_list(guess)
        elif isinstance(guess, np.ndarray):
            guess = to_torch(guess, **self.tensor_args)

            if guess.ndim == 2:
                guess = guess.unsqueeze(0)

            guess_l = [x for x in guess]
            initial_traj_pos_vel = self._handle_guess_list(guess)
        else:
            raise NotImplementedError

        if self.add_noise_before_optimization:
            noise = torch.randn(initial_traj_pos_vel.shape, **self.tensor_args) * self.noise_std
            noise[:, 0, :] = 0
            noise[:, -1, :] = 0
            initial_traj_pos_vel += noise

        return self.optimizer_impl.optimize(guess=initial_traj_pos_vel, **kwargs)

    def _handle_guess_list(self, traj_l):
        traj_pos_l = []

        for traj in traj_l:
            # If no solution was found, create a linear interpolated trajectory between start and finish, even
            # if is not collision-free
            if traj is None:
                # traj = tensor_linspace_v1(
                #     self.sample_based_planner.start_state_pos, self.sample_based_planner.goal_state_pos,
                #     steps=self.optimizer_impl.n_support_points
                # ).T
                raise NotImplementedError()

            traj = traj.squeeze()

            traj_pos, traj_vel = smoothen_trajectory(
                traj, n_support_points=self.optimizer_impl.n_support_points, dt=self.optimizer_impl.dt,
                set_average_velocity=True, tensor_args=self.tensor_args
            )
            # Reshape for gpmp/sgpmp interface
            # initial_traj_pos_vel = torch.cat((traj_pos, traj_vel), dim=-1)
            initial_traj_pos = traj_pos

            # traj_pos_vel_l.append(initial_traj_pos_vel)
            traj_pos_l.append(initial_traj_pos)

        # initial_traj_pos_vel = torch.stack(traj_pos_vel_l)

        try:
            initial_traj_pos = torch.stack(traj_pos_l)
        except RuntimeError:
            breakpoint()
            pass

        return initial_traj_pos
