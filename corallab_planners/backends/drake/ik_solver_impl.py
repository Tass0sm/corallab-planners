import numpy as np

from pydrake.multibody.inverse_kinematics import GlobalInverseKinematics
from pydrake.common.eigen_geometry import Quaternion
import pydrake.solvers as mp

from corallab_lib import Robot, InverseKinematicsProblem

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from corallab_planners.multi_processing import MultiProcessor
# from ..planner_interface import PlannerInterface


class DrakeIKSolver():

    def __init__(
            self,
            solver_name : str,
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

        self.solver_name = solver_name
        self.problem = problem

        robot_id = problem.robot.id
        drake_robot = Robot(robot_id, backend="drake")

        self.global_ik = GlobalInverseKinematics(drake_robot.plant)

        for name, pose in problem.goal_poses.items():
            body = drake_robot.plant.GetBodyByName(name)
            np_pose = pose.position[0].unsqueeze(1).cpu().numpy()
            lb_tol = np.ones_like(np_pose) * 0.05
            ub_tol = np.ones_like(np_pose) * 0.05

            np_quat = pose.quaternion[0].unsqueeze(1).cpu().numpy()
            quat = Quaternion(np_quat)

            self.global_ik.AddWorldPositionConstraint(body.index(), np_pose, lb_tol, ub_tol)
            self.global_ik.AddWorldOrientationConstraint(body.index(), quat, 0.1)

        np_retract_config = problem.retract_config.unsqueeze(1).cpu().numpy()
        pos_cost = np.ones((drake_robot.plant.num_bodies(), 1)) * 1
        orn_cost = np.ones((drake_robot.plant.num_bodies(), 1)) * 1

        self.global_ik.AddPostureCost(
            np_retract_config,
            pos_cost,
            orn_cost,
        )

# q_desired: numpy.ndarray[numpy.float64[m, 1]],
#                                     body_position_cost: numpy.ndarray[numpy.float64[m, 1]],
#                                     body_orientation_cost: numpy.ndarray[numpy.float64[m, 1]])


        # self.q_dim = problem.get_q_dim()

        # self.allowed_time = allowed_time
        # self.simplify_solution = simplify_solution
        # self.interpolate_solution = interpolate_solution
        # self.interpolate_num = interpolate_num
        # self.seed = seed

        # # OMPL Objects
        # self.space = StateSpace(self.q_dim)

        # min_q_bounds = (problem.get_q_min() * 2).tolist()
        # max_q_bounds = (problem.get_q_max() * 2).tolist()
        # bounds = ob.RealVectorBounds(self.q_dim)
        # joint_bounds = zip(min_q_bounds, max_q_bounds)
        # for i, (lower_limit, upper_limit) in enumerate(joint_bounds):
        #     bounds.setLow(i, lower_limit)
        #     bounds.setHigh(i, upper_limit)
        # self.space.setBounds(bounds)

        # self.ss = og.SimpleSetup(self.space)
        # self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        # self.si = self.ss.getSpaceInformation()

        # if ValidStateSamplerOverride:
        #     def allocValidStateSampler(si):
        #         return ValidStateSamplerOverride(si, **sampler_kwargs)

        #     self.si.setValidStateSamplerAllocator(
        #         ob.ValidStateSamplerAllocator(allocValidStateSampler)
        #     )

        # if self.simplify_solution:
        #     self.ps = og.PathSimplifier(self.si)

        # self.planner_name = planner_name
        # self.set_planner(planner_name)
        pass

    @property
    def name(self):
        return f"drake_{self.solver_name}"

    # def set_planner(self, planner_name):
    #     if planner_name == "PRM":
    #         self.planner = og.PRM(self.ss.getSpaceInformation())
    #     elif planner_name == "RRT":
    #         self.planner = og.RRT(self.ss.getSpaceInformation())
    #     elif planner_name == "RRTConnect":
    #         self.planner = og.RRTConnect(self.ss.getSpaceInformation())
    #     elif planner_name == "RRTstar":
    #         self.planner = og.RRTstar(self.ss.getSpaceInformation())
    #     elif planner_name == "EST":
    #         self.planner = og.EST(self.ss.getSpaceInformation())
    #     elif planner_name == "FMT":
    #         self.planner = og.FMT(self.ss.getSpaceInformation())
    #     elif planner_name == "BITstar":
    #         self.planner = og.BITstar(self.ss.getSpaceInformation())
    #     elif planner_name == "STRIDE":
    #         self.planner = og.STRIDE(self.ss.getSpaceInformation())
    #     elif planner_name == "AITStar":
    #         self.planner = og.AITstar(self.ss.getSpaceInformation())
    #     elif planner_name == "KPIECE1":
    #         self.planner = og.KPIECE1(self.ss.getSpaceInformation())
    #     else:
    #         print("{} not recognized, please add it first".format(planner_name))
    #         return

    #     if planner_name not in ["PRM", "AITStar"]:
    #         self.planner.setRange(1.0)

    #     self.ss.setPlanner(self.planner)

    def solve(
            self,
            **kwargs,
    ):

        solver = mp.IpoptSolver()

        breakpoint()

        if solver.available():
            np_retract_config = self.problem.retract_config.unsqueeze(1).cpu().numpy()
            self.global_ik.SetInitialGuess(q=np_retract_config)
            result = solver.Solve(self.global_ik.prog())
            # self.assertTrue(result.is_success())
            global_ik.ReconstructGeneralizedPositionSolution(result=result)
        else:
            raise NotImplementedError()


        # breakpoint()

        # solver_id = SolverId("MixedIntegerBranchAndBound")
        # mibnb = MixedIntegerBranchAndBound(self.program.prog(), solver_id)
        # result = mibnb.Solve()

        # result = Solve(self.program.prog(), initial_guess=np_retract_config)


        # print(f"Success? {result.is_success()}")
        # print(result.get_solution_result())
        # q_sol = result.GetSolution(q)
        # print(q_sol)


        raise NotImplementedError()
