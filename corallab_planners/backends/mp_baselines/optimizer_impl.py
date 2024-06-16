import torch
import numpy as np

from corallab_lib.task import Task

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.stomp import STOMP

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch
from torch_robotics.trajectory.utils import smoothen_trajectory

from ..optimizer_interface import OptimizerInterface


mp_baselines_optimizers = {
    "CHOMP": CHOMP,
    "GPMP2": GPMP2,
    "STOMP": STOMP
}


class MPBaselinesOptimizer(OptimizerInterface):

    def __init__(
            self,
            optimizer_name : str,
            task : Task = None,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        q_dim = task.get_q_dim()
        tmp_start_state = torch.zeros((2 * q_dim,))
        tmp_start_state_pos = torch.zeros((q_dim,))
        tmp_goal_state = torch.zeros((2 * q_dim,))
        tmp_goal_state_pos = torch.zeros((q_dim,))

        task_impl = task.task_impl.task_impl

        OptimizerClass = mp_baselines_optimizers[optimizer_name]

        self.optimizer_impl = OptimizerClass(
            task=task_impl,

            start_state=tmp_start_state,
            start_state_pos=tmp_start_state_pos,

            goal_state=tmp_goal_state,
            goal_state_pos=tmp_goal_state_pos,
            multi_goal_states=tmp_goal_state.unsqueeze(0),

            pos_only=False,
            tensor_args=tensor_args,
            **kwargs,
        )

    @property
    def name(self):
        return f"mp_baselines_{self.optimizer_impl.name}"

    def optimize(
            self,
            guess=None,
            opt_iters=1,
            **kwargs
    ):
        guess = to_torch(guess, **self.optimizer_impl.tensor_args)

        traj_pos, traj_vel = smoothen_trajectory(
            guess[0, 0],
            n_support_points=self.optimizer_impl.n_support_points,
            dt=self.optimizer_impl.dt,
            set_average_velocity=True,
            tensor_args=self.optimizer_impl.tensor_args
        )
        # Reshape for gpmp/sgpmp interface
        initial_traj_pos_vel = torch.cat((traj_pos, traj_vel), dim=-1)
        initial_traj_pos_vel = initial_traj_pos_vel.unsqueeze(0)

        self.optimizer_impl.reset(initial_particle_means=initial_traj_pos_vel)
        soln = self.optimizer_impl.optimize(opt_iters=opt_iters, **kwargs)

        return soln.unsqueeze(0)
