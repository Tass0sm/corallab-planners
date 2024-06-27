import numpy as np

from corallab_lib.task import Task

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from corallab_planners.multi_processing import MultiProcessor
from ..planner_interface import PlannerInterface


DEFAULT_PLANNING_TIME = 10.0


class StateSpace(ob.RealVectorStateSpace):
    def __init__(self, num_dim) -> None:
        super().__init__(num_dim)
        self.num_dim = num_dim
        self.state_sampler = None

    def allocStateSampler(self):
        '''
        This will be called by the internal OMPL planner
        '''
        # WARN: This will cause problems if the underlying planner is multi-threaded!!!
        if self.state_sampler:
            return self.state_sampler

        # when ompl planner calls this, we will return our sampler
        return self.allocDefaultStateSampler()

    def set_state_sampler(self, state_sampler):
        '''
        Optional, Set custom state sampler.
        '''
        self.state_sampler = state_sampler


class OMPLPlanner(PlannerInterface):

    def __init__(
            self,
            planner_name : str,
            task : Task = None,

            allowed_time: float = DEFAULT_PLANNING_TIME,
            simplify_solution: bool = False,
            interpolate_solution: bool = True,
            interpolate_num: int = 64,

            seed : int = 0,

            # Sampler
            # ValidStateSamplerOverride = None,
            # sampler_kwargs = {},
            **kwargs
    ):

        self.task = task

        self.q_dim = task.get_q_dim()

        self.allowed_time = allowed_time
        self.simplify_solution = simplify_solution
        self.interpolate_solution = simplify_solution
        self.interpolate_num = interpolate_num
        self.seed = seed

        # OMPL Objects
        self.space = StateSpace(self.q_dim)

        min_q_bounds = (task.get_q_min() * 2).tolist()
        max_q_bounds = (task.get_q_max() * 2).tolist()
        bounds = ob.RealVectorBounds(self.q_dim)
        joint_bounds = zip(min_q_bounds, max_q_bounds)
        for i, (lower_limit, upper_limit) in enumerate(joint_bounds):
            bounds.setLow(i, lower_limit)
            bounds.setHigh(i, upper_limit)
        self.space.setBounds(bounds)

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        self.si = self.ss.getSpaceInformation()

        # if ValidStateSamplerOverride:
        #     def allocValidStateSampler(si):
        #         return ValidStateSamplerOverride(si, **sampler_kwargs)

        #     self.si.setValidStateSamplerAllocator(
        #         ob.ValidStateSamplerAllocator(allocValidStateSampler)
        #     )

        if self.simplify_solution:
            self.ps = og.PathSimplifier(self.si)

        self.planner_name = planner_name
        self.set_planner(planner_name)

    def set_planner(self, planner_name):
        if planner_name == "PRM":
            self.planner = og.PRM(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation(), addIntermediateStates=True)
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.ss.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.ss.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.ss.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.ss.getSpaceInformation())
        elif planner_name == "STRIDE":
            self.planner = og.STRIDE(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        if planner_name != "PRM":
            self.planner.setRange(1.0)

        self.ss.setPlanner(self.planner)

    def solve(
            self,
            start,
            goal,
            n_trajectories=1,
            **kwargs,
    ):

        info = {}

        # set the start and goal states;
        s = ob.State(self.space)
        g = ob.State(self.space)
        for i in range(len(start)):
            s[i] = start[i].item()
            g[i] = goal[i].item()

        self.ss.setStartAndGoalStates(s, g)

        sol_l = []

        # solve in sequence
        for _ in range(n_trajectories):
            self.reset()

            sol = self._get_single_solution()

            if not self.ss.haveExactSolutionPath():
                print("Did not find exact solution")

            sol_l.append(sol)

        # sols = np.concatenate(sol_l)
        return sol_l, info

    def _get_single_solution(self):
        # attempt to solve the problem within allowed planning time
        solved = self.ss.solve(self.allowed_time)
        sol_path_list = []

        if solved:
            # print("Found solution: interpolating into {} segments".format(INTERPOLATE_NUM))
            # print the path to screen
            sol_path_geometric = self.ss.getSolutionPath()

            if self.interpolate_solution:
                sol_path_geometric.interpolate(self.interpolate_num)

            if self.simplify_solution:
                self.ps.simplify(sol_path_geometric, self.allowed_time)

            if self.interpolate_solution:
                sol_path_geometric.interpolate(self.interpolate_num)

            sol_path_states = sol_path_geometric.getStates()
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            sol_path_arr = np.array(sol_path_list)

            # iters, batches, horizon, q_dim
            sol_path_arr = sol_path_arr.reshape((1, *sol_path_arr.shape))
        else:
            return None

        return sol_path_arr

    def _is_state_valid(self, q):
        q_arr = np.array([q[i] for i in range(self.q_dim)])
        in_collision = self.task.compute_collision(q_arr, margin=0.05).item()
        # if no collision, its valid
        return not bool(in_collision)

    def get_time_used(self):
        return self.ss.getLastPlanComputationTime()

    def state_to_list(self, state):
        return [state[i] for i in range(self.q_dim)]

    def render(self, ax, **kwargs):
        pass

    def reset(self):
        self.ss.clear()
