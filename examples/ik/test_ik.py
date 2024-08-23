import torch
import numpy as np

from corallab_lib import InverseKinematicsProblem, Robot, Pose
from corallab_planners import IKSolver

robot = Robot("DualUR5", backend="curobo")

# goal_poses = {
#     "tool0": Pose(
#         position=torch.tensor([[0.1, 0.1, 0.1]]),
#         quaternion=torch.tensor([[ 0.0000, -0.7071,  0.0000,  0.7071]])
#     )
# }

# "tool0
# link_name = "ee_link"
# goal_poses = {
#     link_name: Pose(
#         position=torch.tensor([[0.2348, 0.1334, 0.5342]]),
#         quaternion=torch.tensor([[ 0.0000, -0.7071,  0.0000,  0.7071]])
#     )
# }


goal_poses = {
    "ee_link_0": Pose(
        position=torch.tensor([[-0.0358,  0.1804,  0.2449],
                               [-0.1448, -0.1173,  0.2389]]),
        quaternion=torch.tensor([[ 0.0000, -0.7071,  0.0000,  0.7071],
                                 [ 0.0000, -0.7071,  0.0000,  0.7071]])
    ),
    "ee_link_1": Pose(
        position=torch.tensor([[-0.0213, -0.1710,  0.2068],
                               [ 0.1134,  0.1886,  0.1769]]),
        quaternion=torch.tensor([[ 0.0000, -0.7071,  0.0000,  0.7071],
                                 [ 0.0000, -0.7071,  0.0000,  0.7071]]),
    )
}

retract_config = torch.tensor(
    [0.0000, -2.2000,  1.9000, -1.3830, -1.5700,  0.0000,
     0.0000, -2.2000,  1.9000, -1.3830, -1.5700,  0.0000]
)

ik_problem = InverseKinematicsProblem(
    robot=robot,
    goal_poses=goal_poses,
    retract_config=retract_config
)

ik_solver = IKSolver(
    problem=ik_problem,
    backend="drake",
)

# fk = goal_poses[link_name]
# print(fk)

result, info = ik_solver.solve()
result = torch.tensor(result).float().unsqueeze(0)
print(result)

fk = robot.differentiable_fk(result)
print(fk)



# Visualize in pybullet

from corallab_lib import Robot, Env, MotionPlanningProblem

cenv = Env(
    "EnvFloor3D",
    ws_limits=np.array([[-1, -1, -0.1],
                        [ 1,  1, 1]]),
    add_plane=True,
    backend="pybullet",
)

# Robot
crobot = Robot(
    "DualUR5",
    backend="pybullet",
)

# Task
problem = MotionPlanningProblem(
    env=cenv,
    robot=crobot,
    backend="pybullet",
)

crobot.set_q(retract_config.squeeze())

input("Continue?")

crobot.set_q(result.squeeze())
