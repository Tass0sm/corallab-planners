import torch
import numpy as np

from corallab_lib import MotionPlanningProblem

from mp_baselines.planners.chomp import CHOMP
from mp_baselines.planners.gpmp2 import GPMP2
from mp_baselines.planners.stomp import STOMP
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite, CostSmoothnessTest, CostSmoothnessCHOMP

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS, to_torch, tensor_linspace_v1
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
            problem : MotionPlanningProblem = None,
            n_support_points : int = 64,
            dt : float = 0.4,
            tensor_args : dict = DEFAULT_TENSOR_ARGS,
            **kwargs
    ):
        assert problem.backend == "torch_robotics", "MPBaselinesPlanner only supports problems using the torch_robotics backend."

        self.n_dof = problem.get_q_dim()
        self.n_support_points = n_support_points
        self.dt = dt

        self.task_impl = problem.problem_impl.task_impl
        self.robot_impl = problem.robot.robot_impl
        self.tensor_args = self.task_impl.tensor_args
        self.optimizer_name = optimizer_name
        self.OptimizerClass = mp_baselines_optimizers[optimizer_name]
        self.config = kwargs
        self.config["n_support_points"] = n_support_points
        self.config["dt"] = dt

        if self.optimizer_name == "CHOMP":
            self.config["pos_only"] = False

    @property
    def name(self):
        return f"mp_baselines_{self.optimizer_name}"

    def optimize(
            self,
            guess=None,
            objective=None,
            **kwargs
    ):
        if isinstance(guess, list):
            initial_traj_pos_vel = self._handle_guess_list(guess)
        elif isinstance(guess, torch.Tensor):
            guess_l = [x for x in guess]
            initial_traj_pos_vel = self._handle_guess_list(guess)
        elif isinstance(guess, np.ndarray):
            guess = to_torch(guess, **self.tensor_args)
            guess_l = [x for x in guess]
            initial_traj_pos_vel = self._handle_guess_list(guess)
        else:
            raise NotImplementedError

        start_state = initial_traj_pos_vel[0, 0, :self.n_dof]
        start_state_pos = initial_traj_pos_vel[0, 0, :self.n_dof]
        goal_state = initial_traj_pos_vel[0, -1, :self.n_dof]
        goal_state_pos = initial_traj_pos_vel[0, -1, :self.n_dof]
        multi_goal_states = goal_state.unsqueeze(0)
        num_particles_per_goal=initial_traj_pos_vel.shape[0]

        self.optimizer_impl = self.OptimizerClass(
            n_dof=self.n_dof,
            task=self.task_impl,
            robot=self.robot_impl,

            start_state=start_state,
            start_state_pos=start_state_pos,

            goal_state=goal_state,
            goal_state_pos=goal_state_pos,
            multi_goal_states=multi_goal_states,

            num_particles_per_goal=num_particles_per_goal,

            cost=objective,
            collision_fields=self.task_impl.get_collision_fields(),

            tensor_args=self.tensor_args,
            **self.config,
            **kwargs,
        )

        if initial_traj_pos_vel.ndim == 2:
            initial_traj_pos_vel = initial_traj_pos_vel.unsqueeze(0).unsqueeze(0)
        elif initial_traj_pos_vel.ndim == 3:
            initial_traj_pos_vel = initial_traj_pos_vel.unsqueeze(0)

        self.optimizer_impl.reset(initial_particle_means=initial_traj_pos_vel)

        trajs_0 = self.optimizer_impl.get_traj()
        trajs_iters = torch.empty((self.optimizer_impl.opt_iters + 1, *trajs_0.shape), **self.optimizer_impl.tensor_args)
        trajs_iters[0] = trajs_0

        for i in range(self.optimizer_impl.opt_iters):
            trajs = self.optimizer_impl.optimize(opt_iters=1, **kwargs)
            trajs_iters[i + 1] = trajs

        info = {"solution_iters": trajs_iters}

        return trajs, info

    def _handle_guess_list(self, traj_l):
        traj_pos_vel_l = []

        for traj in traj_l:
            # If no solution was found, create a linear interpolated trajectory between start and finish, even
            # if is not collision-free
            if traj is None:
                traj = tensor_linspace_v1(
                    self.sample_based_planner.start_state_pos, self.sample_based_planner.goal_state_pos,
                    steps=self.n_support_points
                ).T

            traj = traj.squeeze()

            traj_pos, traj_vel = smoothen_trajectory(
                traj, n_support_points=self.n_support_points, dt=self.dt,
                set_average_velocity=True, tensor_args=self.tensor_args
            )
            # Reshape for gpmp/sgpmp interface
            initial_traj_pos_vel = torch.cat((traj_pos, traj_vel), dim=-1)

            traj_pos_vel_l.append(initial_traj_pos_vel)

        initial_traj_pos_vel = torch.stack(traj_pos_vel_l)
        return initial_traj_pos_vel
