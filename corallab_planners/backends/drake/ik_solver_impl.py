import numpy as np

from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.common.eigen_geometry import Quaternion
from pydrake.multibody.parsing import Parser
from pydrake.systems.analysis import Simulator
from pydrake.all import MultibodyPlant

from pydrake.solvers import MathematicalProgram, Solve

from corallab_lib import Robot, InverseKinematicsProblem

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from corallab_planners.multi_processing import MultiProcessor
# from ..planner_interface import PlannerInterface


class DrakeIKSolver():

    def __init__(
            self,
            # solver_name : str,
            problem : InverseKinematicsProblem = None,

            # allowed_time: float = DEFAULT_PLANNING_TIME,
            # simplify_solution: bool = False,
            # interpolate_solution: bool = True,
            # interpolate_num: int = 64,

            # seed : int = 0,

            # Sampler
            # ValidStateSamplerOverride = None,
            # sampler_kwargs = {},
            **kwargs
    ):
        self.solver_name = "InverseKinematics"
        self.problem = problem
        self.retract_config = problem.retract_config.flatten().cpu().numpy()

        # Get Plant
        robot_id = self.problem.robot.id
        self.drake_robot = Robot(robot_id, backend="drake")
        self.plant_f = self.drake_robot.plant
        self.model_idx = self.drake_robot.model_idx
        self.world_frame = self.plant_f.world_frame()

        # Allocate float context to be used by evaluators.
        self.context_f = self.plant_f.CreateDefaultContext()
        # # Create AutoDiffXd plant and corresponding context.
        self.plant_ad = self.plant_f.ToAutoDiffXd()
        self.context_ad = self.plant_ad.CreateDefaultContext()

    @property
    def name(self):
        return f"drake_{self.solver_name}"

    def _add_retract_config_cost(self, prog, variables):
        def distance_to_retract_config(q):
            """Evaluates squared distance between q and retract_q."""
            # Choose plant and context based on dtype.
            if q.dtype == float:
                plant = self.plant_f
                context = self.context_f
            else:
                # Assume AutoDiff.
                plant = self.plant_ad
                context = self.context_ad

            diffs = (q - self.retract_config) + 0.01
            weighted_diffs = diffs
            return np.linalg.norm(weighted_diffs)

        prog.AddCost(distance_to_retract_config, vars=variables)

    def _add_arm_shape_constraints(self, prog, variables):
        # prog.AddConstraint(variables[1] >= -1.9)
        # prog.AddConstraint(variables[1] <= -0.0)
        # prog.AddConstraint(variables[1] >= -1.0)
        # prog.AddCost(distance_to_retract_config, vars=variables)
        pass

    def _add_position_constraint(self, ik, link, pos):
        link_frame = self.plant_f.GetBodyByName(link).body_frame()

        p_BQ = np.array([0.0, 0.0, 0.0])
        # p_AQ_lower = pos - np.array([0.05, 0.05, 0.05])
        # p_AQ_upper = pos + np.array([0.05, 0.05, 0.05])
        p_AQ_lower = pos - np.array([0.2, 0.2, 0.2])
        p_AQ_upper = pos + np.array([0.2, 0.2, 0.2])

        ik.AddPositionConstraint(
            frameB=link_frame, p_BQ=p_BQ,
            frameA=self.world_frame, p_AQ_lower=p_AQ_lower, p_AQ_upper=p_AQ_upper
        )

    def _add_orientation_constraint(self, ik):
        ee_link_frame = self.plant_f.GetBodyByName("ee_link").body_frame()

        identity_quat = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
        ee_target_quat = Quaternion(np.array([ 0.0000, -0.7071,  0.0000,  0.7071]))

        identity_rotation = RotationMatrix(identity_quat)
        ee_target_rotation = RotationMatrix(ee_target_quat)

        # Constrain the identity rotation in the ee_link frame to match the
        # desired rotation in the base frame.
        ik.AddOrientationConstraint(
            frameBbar=ee_link_frame, R_BbarB=identity_rotation,
            frameAbar=self.world_frame, R_AbarA=ee_target_rotation,
            theta_bound=0.05
        )

    def _add_flat_link_constraint(self, ik, link):
        link_frame = self.plant_f.GetBodyByName(link).body_frame()

        z_vec = np.array([[1.0, 0.0, 0.0]]).T
        ee_target_vec = np.array([[0.0, 0.0, -1.0]]).T

        ik.AddAngleBetweenVectorsConstraint(
            frameB=link_frame, nb_B=z_vec,
            frameA=self.world_frame, na_A=ee_target_vec,
            angle_lower=0.0, angle_upper=0.01
        )

    def solve(
            self,
            **kwargs,
    ):
        ik = InverseKinematics(plant=self.plant_f, plant_context=self.context_f)
        for name, pose in self.problem.goal_poses.items():
            pos = pose.position[0].flatten().cpu().numpy()
            self._add_position_constraint(ik, name, pos)
            self._add_flat_link_constraint(ik, name)

        # CUSTOM
        prog = ik.get_mutable_prog()
        q = ik.q()
        self._add_retract_config_cost(prog, q)

        q0 = self.retract_config.astype("float64")
        prog = ik.prog()
        result = Solve(prog, initial_guess=q0)

        q_sol = result.GetSolution(q)
        return q_sol, {
            "success": result.is_success(),
            "result": result.get_solution_result()
        }
