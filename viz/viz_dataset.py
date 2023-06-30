import matplotlib.pyplot as plt
import numpy as np
import torch


def viz_angs(angs: torch.Tensor, height=256, width=256) -> np.array:
    '''
    Args:
        angs: Row matrix for element-wise angles (N, 3)
              [i, j, ang (rad)]

    Returns:
        (dir_x, dir_y): Dense x-y direction matrices
    '''
    dir_x = np.zeros((height, width))
    dir_y = np.zeros((height, width))

    max_num_angs = angs.shape[0]
    for idx in range(max_num_angs):

        i, j, ang = angs[idx]
        i = int(i.item())
        j = int(j.item())
        ang = ang.item()

        # Negative entries [-1, -1, -1] means end of list
        if i < 0:
            break

        dx = np.cos(ang)
        dy = np.sin(ang)

        dir_x[i, j] = dx
        dir_y[i, j] = dy

    return dir_x, dir_y


def viz_dataset_sample(x: torch.tensor,
                       x_hat: torch.tensor,
                       label: dict,
                       file_path: str = None,
                       viz_gt_lanes: bool = False):
    '''
    Args:
        inputs: (2, H, W)
        labels: (3, H, W)
    '''

    dir_x_full, dir_y_full = viz_angs(label['angs_full'])

    fig = plt.gcf()
    fig.set_size_inches(20, 15)

    cols = 4
    rows = 3

    if viz_gt_lanes:
        dir_x_gt, dir_y_gt = viz_angs(label['gt_angs'])
        rows += 1

    plt.subplot(rows, cols, 1)
    plt.imshow(x[0].numpy())
    plt.subplot(rows, cols, 2)
    plt.imshow(x[1].numpy())
    plt.subplot(rows, cols, 3)
    plt.imshow(x[2:5].numpy().transpose(1, 2, 0))
    plt.subplot(rows, cols, 4)
    plt.imshow(x[0].numpy() + 2 * label['traj_present'].numpy(),
               vmin=0,
               vmax=2)

    plt.subplot(rows, cols, 5)
    plt.imshow(x_hat[0].numpy())
    plt.subplot(rows, cols, 6)
    plt.imshow(x_hat[1].numpy())
    plt.subplot(rows, cols, 7)
    plt.imshow(x_hat[2:5].numpy().transpose(1, 2, 0))
    plt.subplot(rows, cols, 8)
    plt.imshow(x_hat[0].numpy() + 2 * label['traj_full'].numpy(),
               vmin=0,
               vmax=2)

    plt.subplot(rows, cols, 9)
    plt.imshow(label['dynamic'].numpy())
    plt.subplot(rows, cols, 11)
    plt.imshow(dir_x_full, vmin=-1, vmax=1)
    plt.subplot(rows, cols, 12)
    plt.imshow(dir_y_full, vmin=-1, vmax=1)

    if viz_gt_lanes:
        plt.subplot(rows, cols, 14)
        plt.imshow(x_hat[0].numpy() + 2 * label['gt_lanes'].numpy(),
                   vmin=0,
                   vmax=2)
        plt.subplot(rows, cols, 15)
        plt.imshow(dir_x_gt, vmin=-1, vmax=1)
        plt.subplot(rows, cols, 16)
        plt.imshow(dir_y_gt, vmin=-1, vmax=1)

    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
        plt.clf()
        plt.close()
    else:
        plt.show()
