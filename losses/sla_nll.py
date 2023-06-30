import torch


def ce(output, label, eps=1e-14):
    '''Cross-entropy term.
    '''
    return -label * torch.log(output + eps)


def eval_sla_nll(output, label, drivable_N):
    '''
    '''
    nll = ce(output, label) + ce(1 - output, 1 - label)
    nll = torch.sum(nll, dim=(1, 2, 3))
    # Mean over 'drivable' elems
    nll = torch.div(nll, drivable_N)
    # Mean over batch dim
    nll = torch.mean(nll)

    return nll
