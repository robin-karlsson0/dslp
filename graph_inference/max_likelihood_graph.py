import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si
from numpy.random import default_rng


def pnt_dist(pose_0: np.array, pose_1: np.array):
    '''
    Returns the Euclidean distance between two poses.
        dist = sqrt( dx**2 + dy**2 )

    Args:
        pose_0: 1D vector [x, y]
        pose_1:
    '''
    dist = np.sqrt(np.sum((pose_1 - pose_0)**2))
    return dist


def bspline(cv, n=100, degree=3):
    """
    Calculate n samples on a bspline

    Ref: https://stackoverflow.com/questions/28279060/splines-with-python-using-control-knots-and-endpoints

    Args:
        cv:     Array ov control vertices
        n:      Number of samples to return
        degree: Curve degree
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree, 1, count - 1)

    # Calculate knot vector
    kv = np.array([0] * degree + list(range(count - degree + 1)) +
                  [count - degree] * degree,
                  dtype='int')

    # Calculate query range
    u = np.linspace(0, (count - degree), n)

    # Calculate result
    return np.array(si.splev(u, (kv, cv.T, degree))).T


def bspline_equidistance(cv, dist=10, n=100, degree=3):
    '''
    '''
    spline = bspline(cv, n, degree)

    # Resample equidistant spline
    # Compute path length
    ds = []
    for idx in range(spline.shape[0] - 1):
        pnt_0 = spline[idx]
        pnt_1 = spline[idx + 1]
        d = pnt_dist(pnt_0, pnt_1)
        ds.append(d)
    path_length = np.sum(ds)

    # Number of pnts
    num_pnts = int(path_length // dist)

    # Assuming that 'data' is rows x dims (where dims is the dimensionality)
    # Ref: https://stackoverflow.com/questions/19117660/how-to-generate-equispaced-interpolating-values
    data = spline
    diffs = data[1:, :] - data[:-1, :]
    dist = np.linalg.norm(diffs, axis=1)
    u = np.cumsum(dist)
    u = np.hstack([[0], u])
    t = np.linspace(0, u[-1], num_pnts)
    resampled = si.interpn((u, ), data, t)

    return resampled


def find_connected_entry_exit_pairs(entry_paths, connecting_paths, exit_paths):

    connected_pnts = []
    for entry_idx, entry_path in enumerate(entry_paths):

        entry_pnt_0, entry_pnt_1 = entry_path
        print('entry')
        print(entry_idx, entry_pnt_0)
        print(entry_idx, entry_pnt_1)

        for con_idx, connecting_path in enumerate(connecting_paths):

            connecting_pnt_0, connecting_pnt_1 = connecting_path
            print('\t connecting')
            print('\t', con_idx, connecting_pnt_0)
            print('\t', con_idx, connecting_pnt_1)

            if entry_pnt_1 == connecting_pnt_0:

                for exit_idx, exit_path in enumerate(exit_paths):

                    # NOTE Mixed indices
                    exit_pnt_0, exit_pnt_1 = exit_path

                    print('\t\t exit')
                    print('\t\t', exit_idx, exit_pnt_0)
                    print('\t\t', exit_idx, exit_pnt_1)

                    if connecting_pnt_1 == exit_pnt_0:

                        connected_pnts.append((entry_pnt_0, exit_pnt_1))
                        print('\t\t\t Connected', entry_pnt_0, exit_pnt_1)

    return connected_pnts


def preproc_entry_exit_pairs(pnt_pairs,
                             pnt_scalar=2,
                             img_frame_dim=256,
                             lane_width_px=12):

    lane_width_px = pnt_scalar * lane_width_px

    pnt_pairs_preproc = []

    for pnt_pair in pnt_pairs:

        entry_i = pnt_pair[0][0]
        entry_j = pnt_pair[0][1]
        exit_i = pnt_pair[1][0]
        exit_j = pnt_pair[1][1]

        entry_i *= pnt_scalar
        entry_j *= pnt_scalar
        exit_i *= pnt_scalar
        exit_j *= pnt_scalar

        entry_i = int(entry_i)
        entry_j = int(entry_j)
        exit_i = int(exit_i)
        exit_j = int(exit_j)

        dist = np.sqrt((exit_i - entry_i)**2 + (exit_j - entry_j)**2)
        if dist < 2 * lane_width_px:
            continue

        # Check if both 'left'
        # if entry_i < 10 and exit_i < 10:
        #     # print('left')
        #     continue
        # Check if both 'bottom'
        # if entry_j > (img_frame_dim - 10) and exit_j > (img_frame_dim - 10):
        #     # print('bottom')
        #     continue
        # Check if both 'bottom'
        # print(entry_i, img_frame_dim - 10)
        # print(exit_i, img_frame_dim - 10)
        # if entry_i > (img_frame_dim - 10) and exit_i > (img_frame_dim - 10):
        #     # print('right')
        #     continue
        # Check if both 'left'
        # if entry_j < 10 and exit_j < 10:
        #     # print('top')
        #     continue

        pnt_pairs_preproc.append(([entry_i, entry_j], [exit_i, exit_j]))

    return pnt_pairs_preproc


def path_to_mask(path: np.array, height: int, width: int):
    '''
    Converts a set of path points to a boolean mask.

    Args:
        path: Matrix (N,2) of (i, j) image coordinates (int).
        height: Height of image frame.
        width: Width of image frame.

    Returns:
        Boolean mask of size (height, width) with 'True' values for path points.

    '''
    path_mask = np.zeros((256, 256), dtype=bool)
    path_mask[path[:, 1], path[:, 0]] = True

    return path_mask


def path_nll_da(path, da, eps=1e-24):
    '''
    Compute the negative log likelihood of path given predicted directional affordance.

    Args:
        path: Matrix (N,2) of (i, j) image coordinates (int).
        da: Tensor (D,H,W) with directional affordance probabilty values.

    Returns:
        Mean NLL of path.
    '''
    num_dirs, height, width = da.shape
    path_mask = path_to_mask(path, height, width)

    delta_phi = 2. * np.pi / num_dirs

    nll = 0
    for idx in range(path.shape[0] - 1):
        # Compute angle to next point
        i0 = path[idx, 0]
        j0 = path[idx, 1]

        i1 = path[idx + 1, 0]
        j1 = path[idx + 1, 1]

        di = i1 - i0
        dj = j1 - j0

        ang = np.arctan2(di, dj) - 0.5 * np.pi
        if ang < 0:
            ang += 2. * np.pi

        # Get idx of corresponding probability of angle interval
        # Ex: 23.456 // 10 := 2
        ang_idx = int(ang // delta_phi)

        # Prob of path direction
        # Apply convolution to allow diagonal transitions in straight
        # directional fields
        #p_dir = da[:, j0, i0]
        #kernel = np.array([0.25, 0.50, 0.25])
        #p_dir_padding = np.pad(p_dir, 1, 'wrap')
        #p_dir_padding = np.convolve(p_dir_padding, kernel, mode='same')
        #p_dir = p_dir_padding[1:-1]
        #prob = p_dir[ang_idx]

        prob = da[ang_idx, j0, i0]

        nll += -1 * np.log(prob + eps)

    return nll  # / np.sum(path_mask)


def path_nll_sla(path: np.array, sla: np.array, eps=1e-24):
    '''
    Compute the negative log likelihood of path given predicted soft lane affordance.

    Args:
        path: Matrix (N,2) of (i, j) image coordinates (int).
        sla: Matrix (H,W) with soft lane affordance probabilty values.

    Returns:
        Mean NLL of path.
    '''
    height, width = sla.shape
    path_mask = path_to_mask(path, height, width)

    if (sla[path_mask] == 0).any():
        return np.inf

    nll = -1 * np.log(path_mask * sla + eps)
    nll = path_mask * nll

    return np.sum(nll)  # / np.sum(path_mask)


def find_max_likelihood_path(entry_i,
                             entry_j,
                             exit_i,
                             exit_j,
                             sla,
                             da,
                             num_samples=1000,
                             num_pnts=50,
                             nll_da_weight=1,
                             img_frame_dim=256,
                             sampling_distr_ratio=0.6):
    '''
    '''
    rng = default_rng()

    half_dist = 0.5 * img_frame_dim
    var = (half_dist * sampling_distr_ratio)**2
    vals = rng.multivariate_normal((half_dist, half_dist),
                                   cov=var * np.eye(2),
                                   size=num_samples)

    best_path = None
    best_nll_sla = None
    best_nll_da = None
    best_cv = None
    best_nll = np.inf
    paths = []

    for idx in range(num_samples):

        val = vals[idx]

        # Constrain path to be within image frame
        if (val < 0).any() or (val >= 256).any():
            continue

        cv_i = val[0]
        cv_j = val[1]

        cv = np.array([[entry_j, entry_i], [cv_j, cv_i], [exit_j, exit_i]])

        # path = bspline(cv, n=num_pnts, degree=2)
        path = bspline_equidistance(cv, dist=10, n=num_pnts, degree=2)
        path = path.astype(int)
        paths.append(path)

        nll_sla = path_nll_sla(path, sla)
        nll_da = path_nll_da(path, da)
        nll = nll_sla + nll_da_weight * nll_da

        if nll < best_nll:
            best_cv = cv
            best_path = path
            best_nll_sla = nll_sla
            best_nll_da = nll_da
            best_nll = nll

    return best_path, best_nll_sla, best_nll_da, best_cv, paths


def find_max_likelihood_paths(pnt_pairs,
                              sla,
                              da,
                              num_samples=1000,
                              num_pnts=50):
    '''
    '''
    paths = []
    for pnt_pair in pnt_pairs:

        entry_j = pnt_pair[0][0]
        entry_i = pnt_pair[0][1]
        exit_j = pnt_pair[1][0]
        exit_i = pnt_pair[1][1]

        best_path, best_nll_sla, best_nll_da, _, _ = find_max_likelihood_path(
            entry_i,
            entry_j,
            exit_i,
            exit_j,
            sla,
            da,
            num_samples=num_samples,
            num_pnts=num_pnts)

        # No connecting polynomial found
        if best_path is None:
            continue

        paths.append(best_path)

    return paths


def find_max_likelihood_graph(sla,
                              da,
                              entry_paths,
                              con_paths,
                              exit_paths,
                              num_samples=1000,
                              num_pnts=50):
    '''
    '''
    # NOTE Temporary function for all entry to all exit connectivity
    pnt_pairs = []
    for entry_path in entry_paths:
        for exit_path in exit_paths:
            pnt_pairs.append([entry_path[0], exit_path[1]])

    pnt_pairs = preproc_entry_exit_pairs(pnt_pairs)

    paths = find_max_likelihood_paths(pnt_pairs,
                                      sla,
                                      da,
                                      num_samples=num_samples,
                                      num_pnts=num_pnts)

    return paths


if __name__ == '__main__':

    with open('sample_12.pkl', 'rb') as file:
        sample = pickle.load(file)

    sla = sample['sla']
    da = sample['da']
    entry_paths = sample['entry_paths']
    connecting_paths = sample['connecting_pnts']
    exit_paths = sample['exit_paths']

    # TODO Fix inexact DAG points
    # pnt_pairs = find_connected_entry_exit_pairs(entry_paths, connecting_paths, exit_paths)

    # NOTE Temporary function for all entry to all exit connectivity
    pnt_pairs = []
    for entry_path in entry_paths:
        for exit_path in exit_paths:
            pnt_pairs.append([entry_path[0], exit_path[1]])

    for pnts in pnt_pairs:
        print(pnts)

    plt.imshow(cv2.resize(sla, (128, 128), interpolation=cv2.INTER_LINEAR))
    plt.show()

    pnt_pairs = preproc_entry_exit_pairs(pnt_pairs)

    for pnts in pnt_pairs:
        print(pnts)

    paths = find_max_likelihood_paths(pnt_pairs, sla, da)

    plt.imshow(sla)
    for path in paths:
        i, j = path.T
        plt.plot(i, j, 'k-')
        di = i[-1] - i[-2]
        dj = j[-1] - j[-2]
        plt.arrow(i[-2],
                  j[-2],
                  di,
                  dj,
                  head_width=5,
                  facecolor='k',
                  length_includes_head=True)
    plt.show()
