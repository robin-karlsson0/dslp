import numpy as np
from scipy.signal import argrelextrema


def eval_dir_acc(da: np.array,
                 mm_gt_angs_tensor: np.array,
                 threshold_ang: float = 90) -> float:
    '''
    Args:
        da: Predicted directional probabilities (D, H, W).
        gt_lane: Lanes plotted onto a dense boolean map (H, W).
        mm_gt_angs_tensor: GT lane graph encoded directional probabilities
                           (D, H, W).
        threshold_ang: Predicted direction correct if sufficient probability
                       by integrating this interval.

    Returns:

    '''
    num_dirs, I, J = da.shape

    delta_ang = 360 / num_dirs

    # Index interval i - delta_idx : i + delta_idx corresponding to angle
    delta_idx = int(np.floor(0.5 * (threshold_ang / delta_ang)))

    # Probability of direction interval given uniform probabilities
    p_dir_uniform = 1. / num_dirs
    p_dir_uniform_int = np.sum(2 * delta_idx * p_dir_uniform)

    # Find indices of elements having a GT direction encoded

    have_dirs = np.max(mm_gt_angs_tensor, axis=0) > p_dir_uniform
    i_idxs, j_idxs = np.where(have_dirs)

    # List with boolean values for correct direction
    dir_in_thresh = []

    for idx in range(len(i_idxs)):

        i = i_idxs[idx]
        j = j_idxs[idx]

        p_dir_pred = da[:, i, j]
        p_dir_gt = mm_gt_angs_tensor[:, i, j]

        # Find GT directions as local maximums
        # NOTE: Perturbs values to avoid plateaus
        p_dir_gt += 1e-5 * np.arange(0, len(p_dir_gt))
        p_dir_gt[p_dir_gt < p_dir_uniform] = 0
        gt_dir_idxs = argrelextrema(p_dir_gt, np.greater, mode='wrap')

        for gt_dir_idx in gt_dir_idxs:
            gt_dir_idx = gt_dir_idx[0]

            # Sum predicted directional probabilities within threshold angle
            idx_0 = gt_dir_idx - delta_idx
            idx_1 = gt_dir_idx + delta_idx
            p_dir_pred_idxs = np.arange(idx_0, idx_1, 1)

            # Reflect undershoot/overshoot idxs to other side of cyclical range
            mask = p_dir_pred_idxs < 0
            p_dir_pred_idxs[mask] = p_dir_pred_idxs[mask] + num_dirs
            mask = p_dir_pred_idxs >= num_dirs
            p_dir_pred_idxs[mask] = p_dir_pred_idxs[mask] - num_dirs

            p_dir_pred_int = p_dir_pred[p_dir_pred_idxs]
            p_dir_pred_int = np.sum(p_dir_pred_int)

            if p_dir_pred_int > p_dir_uniform_int:
                dir_in_thresh.append(True)
            else:
                dir_in_thresh.append(False)

    acc = np.sum(dir_in_thresh) / len(dir_in_thresh)

    return acc
