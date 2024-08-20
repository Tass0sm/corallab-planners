class Constraint:

    def __init__(self, i, traj, time):
        self.i = i
        self.traj = traj
        self.time = time

class Conflict:

    def __init__(self, i, j, traj_i, traj_j, time):
        self.i = i
        self.j = j
        self.traj_i = traj_i
        self.traj_j = traj_j
        self.time = time

    # def __str__(self):
    #     return "VC: " + str([str(vc) for vc in self.vertex_constraints])  + \
    #         "EC: " + str([str(ec) for ec in self.edge_constraints])

class HighLevelNode:

    def __init__(self, solutions, constraints=set(), parent=None):
        self.solutions = solutions
        self.constraints = constraints
        self.parent = parent
        self.compute_cost()

    def set_plan(self, solutions, times):
        self.solutions = solutions
        self.times = times

    def has_full_solution(self):
        return all([sol is not None for sol in self.solutions.values()])

    def compute_cost(self):
        if self.solutions:
            self.cost = 0

            for sol in self.solutions.values():
                if sol is None:
                    self.cost += 999
                else:
                    self.cost += sol.diff(dim=0).square().sum(dim=-1).sqrt().sum()
        else:
            self.cost = 999.0

    def __eq__(self, other):
        if not isinstance(other, type(self)): return NotImplemented
        return self.solution == other.solution and self.cost == other.cost

    def __hash__(self):
        return hash((self.cost))

    def __lt__(self, other):
        return self.cost < other.cost
