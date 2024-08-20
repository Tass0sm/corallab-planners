import torch
import numpy as np
from heapq import heappop, heappush

from corallab_lib import PebbleMotionProblem
from corallab_planners import Planner


class CBS:

    def __init__(
            self,
            problem : PebbleMotionProblem,
            n_iters: int = 30000,
            max_time: float = 60.,
    ):
        self.problem = problem
        self.graph = problem.graph

        self.low_level_problem = PebbleMotionProblem(
            graph=self.graph,
            n_pebbles=1,
            backend="corallab"
        )

        self.low_level_planner = Planner(
            "SCIPP",
            problem=self.low_level_problem,
            backend="corallab"
        )

    def _solve_individual_problems(self, starts, goals):
        sols = {}

        for i, (start_i, goal_i) in enumerate(zip(starts, goals)):
            breakpoint()

            sol, info = self.low_level_planner.solve(start_i, goal_i, constraints=None)

            # sols[i] = sol
            sols[i] = torch.tensor(sol, **self.problem.tensor_args)

        return sols

    def _get_first_conflict(self, node):
        solutions = node.solutions
        joint_solutions_l = self._create_joint_solution(solutions, concatenate=False)
        joint_solution = torch.cat(joint_solutions_l, axis=-1)

        r_i, r_j = None, None
        t_s, t_e = None, None
        idx_s, idx_e = None, None

        i = 0
        while i < len(joint_solution):
            state = joint_solution[i]
            # in_collision, info = self.problem.get_self_collision_info(state)
            in_collision, info = self.problem.check_collision_info(state, margin=0)

            if info["cost_collision_objects"]:
                print("Somehow found object collision?")
                break

            if info["cost_collision_border"]:
                print("Somehow found border collision?")
                break

            if in_collision:
                idx_s = i

                r_i, r_j = info["self_collision_robots"][0, 1:]
                r_i = r_i.item()
                r_j = r_j.item()

                j = i + 1
                while j < len(joint_solution):
                    state = joint_solution[j]
                    in_collision, info = self.problem.check_collision_info(state, margin=0)

                    if info["self_collision_robots"] is None or info["self_collision_robots"].nelement() == 0:
                        self_collision_robots = []
                    else:
                        self_collision_robots = info["self_collision_robots"][:, 1:].flatten().unique()

                    if r_i not in self_collision_robots and r_j not in self_collision_robots:
                        idx_e = j+1
                        break

                    j += 1

            if t_e is not None:
                break

            i += 1

        if r_i is None:
            return None
        else:
            time = torch.arange(idx_s, idx_e+1)
            traj_i = joint_solutions_l[r_i][idx_s:idx_e+1]
            traj_j = joint_solutions_l[r_j][idx_s:idx_e+1]

            return Conflict(r_i, r_j, traj_i, traj_j, time)

    def _attempt_replan(self, robot_idx, node, start, goal):
        solutions_l = list(node.solutions.values())
        max_length = max([sol.shape[0] for sol in solutions_l])

        unique_subrobot_idx = self.subrobot_to_unique_subrobot_map[robot_idx]
        prm = self.prms[unique_subrobot_idx]

        tmp = node
        constraint_set = set()
        while tmp is not None:
            constraint_set |= tmp.constraints
            tmp = tmp.parent

        # Solve new dynamic problem
        subrobot_starts = self._get_subrobot_states(start)
        subrobot_goals = self._get_subrobot_states(goal)
        start_i = subrobot_starts[robot_idx]
        goal_i = subrobot_goals[robot_idx]

        # TODO: ??????
        sol = constrained_a_star(self.problem, prm.planner_impl.planner_impl, robot_idx, start_i, goal_i, max_length, constraint_set)

        if isinstance(sol, list):
            sol = sol[0]

        if sol is not None:
            node.solutions[robot_idx] = sol
            node.times[robot_idx] = time
        else:
            print("Failed Replanning")
            node.solutions[robot_idx] = None

        node.compute_cost()

    def solve(
            self,
            starts,
            goals,
            **kwargs
    ):
        assert len(starts) == self.problem.n_pebbles, "Number of starts must match number of pebbles"
        assert len(goals) == self.problem.n_pebbles, "Number of goals must match number of pebbles"

        if any(s not in self.graph for s in starts) or any(g not in self.graph for g in goals):
            return None

        # Solve individual problems
        solutions = self._solve_individual_problems(starts, goals)
        ct_root = HighLevelNode(solutions)

        queue = [ct_root]

        iteration = -1
        solution = None

        with TimerCUDA() as t:
            while (t.elapsed < self.max_time) and (iteration < self.n_iters):
                iteration += 1

                current_node = queue[0]
                k = self._get_first_conflict(current_node)

                if k is None:
                    solution = current_node;
                    break;
                else:
                    solution = current_node;
                    break;

                    # current_node = heapq.heappop(queue)

                    # for a_idx, a_traj, b_idx in [(k.i, k.traj_i, k.j),
                    #                              (k.j, k.traj_j, k.i)]:
                    #     new_node_constraints = {Constraint(a_idx, a_traj, k.time)}
                    #     new_node = HighLevelNode(
                    #         copy(current_node.solutions),
                    #         constraints=new_node_constraints,
                    #         parent=current_node
                    #     )

                    #     self._attempt_replan(b_idx, new_node, start, goal)
                    #     heapq.heappush(queue, new_node)

        if solution is None:
            return None, {}

        joint_solution = self._create_joint_solution(solution.solutions, concatenate=True)

        return joint_solution, { }
