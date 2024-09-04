import numpy as np

from corallab_lib import DynamicPlanningProblem

from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og

from corallab_planners.multi_processing import MultiProcessor
from ..planner_interface import PlannerInterface
from .planner_impl import StateSpace


DEFAULT_PLANNING_TIME = 30.0


class SpaceTimeMotionValidator(ob.MotionValidator):
    def __init__(self, si):
        super().__init__(si)
        self.si = si

    def checkMotion(self, s1, s2):
        if not self.si.isValid(s2):
            return False

        delta_pos = self.si.getStateSpace().distanceSpace(s1, s2)
        # delta_t = self.si.getStateSpace().distanceTime(s1, s2)
        t1 = self.si.getStateSpace().getStateTime(s1)
        t2 = self.si.getStateSpace().getStateTime(s2)
        delta_t = t2 - t1

        if delta_t <= 0 :
            return False

        if (delta_pos / delta_t) > self.si.getStateSpace().getVMax():
            return False

        return True

class SpaceTimeStateSpace(ob.SpaceTimeStateSpace):
    # def __init__(self, state_space : StateSpace) -> None:
    #     super().__init__(state_space)

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


class DeterministicSpaceTimeStateSampler(ob.CompoundStateSampler):
    pass
    # def __init__(self):
    #     breakpoint
    #     pass


class OMPLDynamicPlanner(PlannerInterface):

    def __init__(
            self,
            planner_name : str,
            problem : DynamicPlanningProblem = None,

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

        self.problem = problem

        self.q_dim = problem.get_q_dim()

        self.allowed_time = allowed_time
        self.simplify_solution = simplify_solution
        self.interpolate_solution = interpolate_solution
        self.interpolate_num = interpolate_num
        self.seed = seed

        self.t_max = problem.t_max
        self.v_max = problem.v_max

        # OMPL Objects
        state_space = StateSpace(self.q_dim)

        min_q_bounds = problem.get_q_min().tolist()
        max_q_bounds = problem.get_q_max().tolist()
        bounds = ob.RealVectorBounds(self.q_dim)
        joint_bounds = zip(min_q_bounds, max_q_bounds)
        for i, (lower_limit, upper_limit) in enumerate(joint_bounds):
            bounds.setLow(i, lower_limit)
            bounds.setHigh(i, upper_limit)

        state_space.setBounds(bounds)
        # state_space.set_state_sampler(ob.RealVectorDeterministicStateSampler(state_space))

        self.space = SpaceTimeStateSpace(state_space, self.v_max)
        self.space.setTimeBounds(0.0, self.t_max)
        self.space.updateEpsilon()

        self.space.set_state_sampler(DeterministicSpaceTimeStateSampler(self.space))

        self.ss = og.SimpleSetup(self.space)
        self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self._is_state_valid))
        self.si = self.ss.getSpaceInformation()
        self.si.setMotionValidator(SpaceTimeMotionValidator(self.si))

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

    @property
    def name(self):
        return f"ompl_{self.planner_name}"

    def set_planner(self, planner_name):
        if planner_name == "STRRTstar":
            self.planner = og.STRRTstar(self.ss.getSpaceInformation())
        elif planner_name == "RRT":
            self.planner = og.RRT(self.ss.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.ss.getSpaceInformation())
        else:
            print("{} not recognized, please add it first".format(planner_name))
            return

        # if planner_name not in ["PRM", "AITStar"]:
        #     self.planner.setRange(1.0)

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

        # Set the spatial start and goal states
        for i in range(len(start)):
            s[i] = start[i].item()
            g[i] = goal[i].item()

        # Set the temporal start and goal states
        t_idx = len(start)
        s[t_idx] = 0.0
        g[t_idx] = self.t_max

        self.ss.setStartAndGoalStates(s, g)

        sol_l = []
        time_l = []

        # solve in sequence
        for _ in range(n_trajectories):
            self.reset()

            sol, time = self._get_single_solution()

            if not self.ss.haveExactSolutionPath():
                print("Did not find exact solution")

            sol_l.append(sol)
            time_l.append(time)

        if all([x is None for x in sol_l ]):
            info["failed"] = True

        info["time"] = time_l

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
            sol_path_times = np.array([self.space.getStateTime(state) for state in sol_path_states])
            sol_path_list = [self.state_to_list(state) for state in sol_path_states]
            sol_path_arr = np.array(sol_path_list)
        else:
            return None, None

        return sol_path_arr, sol_path_times

    def _is_state_valid(self, compound_state):
        time = self.space.getStateTime(compound_state)
        q = compound_state[0]

        q_arr = np.array([q[i] for i in range(self.q_dim)])
        q_arr = q_arr.reshape((1, 1, -1))
        in_collision = self.problem.check_collision(time, q_arr).item()

        # in_static_collision = self.problem.static_check_collision(q_arr).item()
        # print("in_static_collision", in_static_collision)
        # print("in_collision", in_collision)

        # if no collision, its valid
        return not bool(in_collision)

    def get_time_used(self):
        return self.ss.getLastPlanComputationTime()

    def state_to_list(self, compound_state):
        q = compound_state[0]
        return [q[i] for i in range(self.q_dim)]

    def render(self, ax, **kwargs):
        pass

    def reset(self):
        self.ss.clear()
