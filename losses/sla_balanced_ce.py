import torch


def ce(output, label, eps=1e-14):
    '''Cross-entropy term.
    '''
    return -label * torch.log(output + eps)


def loss_sla_balanced_ce(output, label, alpha, drivable_N):
    '''Computes the 'Soft-Lane Affordance' loss for an output-label tensor pair.

    By removing obvious 'obstacle elements' from the output, the model is able
    to learn about the actual road scene more effectively.

    '''
    # Compute the ratio between 'True' and 'False' label path elements
    label_elements = torch.sum(label.detach(), (1, 2, 3), keepdim=True)
    beta = torch.div(label_elements + 1, drivable_N)  # (batch_n,1,1,1)

    loss = beta * ce(1 - output, 1 - label) + alpha * (1 - beta) * ce(
        output, label)
    loss = torch.sum(loss, dim=(1, 2, 3), keepdim=True)

    loss = torch.div(loss, drivable_N + 1)

    loss = torch.mean(loss)

    # Loss contribution
    loss_neg = beta * ce(1 - output, 1 - label)
    loss_neg = torch.sum(loss_neg, dim=(1, 2, 3), keepdim=True)
    loss_neg = torch.div(loss_neg, drivable_N - label_elements + 1)
    loss_neg = torch.mean(loss_neg)

    loss_pos = alpha * (1 - beta) * ce(output, label)
    loss_pos = torch.sum(loss_pos, dim=(1, 2, 3), keepdim=True)
    loss_pos = torch.div(loss_pos, label_elements + 1)
    loss_pos = torch.mean(loss_pos)

    return loss, loss_neg.item(), loss_pos.item()
