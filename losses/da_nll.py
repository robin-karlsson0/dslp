import torch


def eval_da_nll(out_da, label_da):
    '''
    Args:
        out_da: (batch_n, ang_range_disc, n, n)
        label_da: dimension (batch_n, ang_range_disc, n, n)
    '''
    ##########################
    #  GENERATE TARGET LABEL
    ##########################
    a = torch.sum(label_da, dim=1)  # (B,H,W)
    dir_path_label = ~torch.isclose(a, torch.ones_like(a))  # (B,H,W)
    dir_path_N = torch.sum(dir_path_label, dim=(-2, -1))

    nll = -1 * label_da * torch.log(out_da)
    nll = torch.sum(nll, dim=(1))
    nll = dir_path_label * nll
    # Mean over 'drivable' elems
    nll = torch.sum(nll, dim=(-2, -1))
    nll = torch.div(nll, dir_path_N)
    # Mean over batch dim
    nll = torch.mean(nll)

    return nll
