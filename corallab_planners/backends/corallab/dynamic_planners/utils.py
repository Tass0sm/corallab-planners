import torch


def check_motion(sequence, time, collision_fn):
    in_collision = collision_fn(sequence, time=time)
    idxs_in_collision = torch.argwhere(in_collision)
    if idxs_in_collision.nelement() == 0:
        if sequence.ndim == 1:
            return sequence, time.item()
        elif sequence.ndim == 2:
            return sequence[-1], time[-1].item()
        elif sequence.ndim == 3:
            return sequence[0, -1], time[-1].item()
    else:
        first_idx_in_collision = idxs_in_collision[0].item()
        if first_idx_in_collision == 0:
            # the first point in the sequence is in collision
            return None, None

        # return the point immediate before the one in collision
        if sequence.ndim == 1:
            return sequence, time.item()
        elif sequence.ndim == 2:
            return sequence[first_idx_in_collision-1], time[first_idx_in_collision-1].item()
        elif sequence.ndim == 3:
            return sequence[0, first_idx_in_collision-1], time[first_idx_in_collision-1].item()
