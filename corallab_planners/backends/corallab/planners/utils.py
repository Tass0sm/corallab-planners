import torch


def extend_path(distance_fn, q1, q2, max_step=0.03, max_dist=0.1, tensor_args=None):
    # max_dist must be <= radius of RRT star!
    dist = distance_fn(q1, q2)
    if dist > max_dist:
        q2 = q1 + (q2 - q1) * (max_dist / dist)

    alpha = torch.linspace(0, 1, int(dist / max_step) + 2).to(**tensor_args)  # skip first and last
    q1 = q1.unsqueeze(0)
    q2 = q2.unsqueeze(0)
    extension = q1 + (q2 - q1) * alpha.unsqueeze(1)
    return extension
