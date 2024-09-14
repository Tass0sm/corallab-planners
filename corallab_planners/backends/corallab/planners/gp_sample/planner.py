import torch

from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS

from mp_baselines.planners.costs.factors.gp_factor import GPFactor
from mp_baselines.planners.costs.factors.mp_priors_multi import MultiMPPrior
from mp_baselines.planners.costs.factors.unary_factor import UnaryFactor

from corallab_lib import MotionPlanningProblem


class GPSamplePlanner:

    def __init__(
        self,
        problem : MotionPlanningProblem = None,
        n_support_points : int = 64,
        tensor_args : dict = DEFAULT_TENSOR_ARGS,
        sigma_start_init : float = 0.001,
        sigma_goal_init : float = 0.001,
        sigma_gp_init : float = 0.03,
        **kwargs
    ):
        self.problem = problem
        self.n_dof = problem.get_q_dim()
        self.n_support_points = n_support_points

        self.tensor_args = tensor_args

        self.dt = 0.4
        self.sigma_start_init = sigma_start_init
        self.sigma_goal_init = sigma_goal_init
        self.sigma_gp_init = sigma_gp_init

    @property
    def name(self):
        return f"gp_sample_planner"

    def get_GP_prior(
            self,
            start_K,
            gp_K,
            goal_K,
            state_init,
            particle_means=None,
            goal_states=None,
    ):
        return MultiMPPrior(
            self.n_support_points - 1,
            self.dt,
            self.n_dof * 2,
            self.n_dof,
            start_K,
            gp_K,
            state_init,
            K_g_inv=goal_K,  # NOTE(sasha) Assume same goal Cov. for now
            means=particle_means,
            goal_states=goal_states,
            tensor_args=self.tensor_args,
        )

    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs
    ):

        # set zero velocity for GP prior
        start_state = torch.cat((start, torch.zeros_like(start)), dim=-1).to(**self.tensor_args)
        start_state_pos = start
        goal_state = torch.cat((goal, torch.zeros_like(goal)), dim=-1).to(**self.tensor_args)
        goal_state_pos = goal

        #========= Initialization factors ===============
        self.start_prior_init = UnaryFactor(
            self.n_dof * 2,
            self.sigma_start_init,
            start_state,
            self.tensor_args,
        )

        self.gp_prior_init = GPFactor(
            self.n_dof,
            self.sigma_gp_init,
            self.dt,
            self.n_support_points - 1,
            self.tensor_args,
        )

        self.goal_prior_init = UnaryFactor(
            self.n_dof * 2,
            self.sigma_goal_init,
            goal_state,
            self.tensor_args,
        )

        self._traj_dist = self.get_GP_prior(
            self.start_prior_init.K,
            self.gp_prior_init.Q_inv[0],
            self.goal_prior_init.K,
            start_state,
            goal_states=goal_state.unsqueeze(0),
        )

        particles = self._traj_dist.sample(n_trajectories).to(**self.tensor_args)

        self.traj_dim = particles.shape
        del self._traj_dist  # free memory

        # chop off velocity for uniformity...
        particles = particles[..., :self.n_dof]

        return particles.flatten(0, 1), {}
