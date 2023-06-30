import numpy as np
import scipy.special
import torch


####################
#  DEBUG FUNCTION
####################
def integrate_distribution(dist, dist_range):
    '''Integrate a distribution using the trapezoidal approximation rule.

        Args:
            dist: Distribution values in 1D array.
            dist_range: Distrbution range in 1D array.

        Returns:
            Integration sum as float.
        '''
    N = dist.shape[0]
    integ_sum = 0.0
    for i in range(N - 1):
        partion_range = dist_range[i + 1] - dist_range[i]
        dist_val = dist[i] + dist[i + 1]
        integ_sum += partion_range * dist_val / 2.0

    return integ_sum


def biternion_to_angle(x, y):
    '''Converts biternion tensor representation to positive angle tensor.
    Args:
        x: Biternion 'x' component of shape (batch_n, n, n)
        y: Biternion 'y' component of shape (batch_n, n, n)
    '''
    ang = torch.atan2(y, x)
    # Add 360 deg to negative angle elements
    mask = (ang < 0).float()
    ang = ang + 2.0 * np.pi * mask
    return ang


def loss_da_kl_div(output_da, mm_ang_label):
    ##########################
    #  GENERATE TARGET LABEL
    ##########################
    a = torch.sum(mm_ang_label, dim=1)  # (B,H,W)
    dir_path_label = ~torch.isclose(a, torch.ones_like(a))  # (B,H,W)

    #################
    #  COMPUTE LOSS
    #################

    # Try just maximizing log liklihood?
    KL_div = mm_ang_label * (torch.log(mm_ang_label + 1e-14) -
                             torch.log(output_da + 1e-14))

    # Sum distribution over every element-> dim (batch_n, y, x, 1)
    # num_angs = output_da.shape[1]
    KL_div = torch.sum(KL_div, dim=1)  # * (2.0 * np.pi / num_angs)

    # Zero non-path elements
    KL_div = KL_div * dir_path_label  # [:, 0].unsqueeze(-1)

    # Sum all element losses -> dim (batch_n)
    KL_div = torch.sum(KL_div, dim=(1, 2))

    # Make loss invariant to path length by average element loss
    # - Summing all '1' elements -> dim (batch_n)
    dir_path_label_N = torch.sum(dir_path_label, dim=(1, 2))
    KL_div = torch.div(KL_div, dir_path_label_N + 1)

    # Average of all batch losses to scalar
    KL_div = torch.mean(KL_div)

    return KL_div
