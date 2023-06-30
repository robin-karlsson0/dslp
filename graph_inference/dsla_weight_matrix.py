import cv2
import matplotlib.pyplot as plt
import numpy as np

from graph_inference.grid_map import (get_neighbor_nodes, node_coord2idx,
                                      node_idx2coord)


def smoothen_sla_map(sla_map, sla_threshold=0.1, kernel_size=8, power=8):
    '''Smooth SLA grid map to penalize paths close to border.
    '''
    # sla_map[sla_map >= sla_threshold] = 1.
    # sla_map[sla_map < 1.] = 0.

    # kernel = (kernel_size, kernel_size)
    # sla_map_ = cv2.blur(sla_map, kernel)
    # sla_map = sla_map_ * sla_map

    # sla_map = sla_map**power

    return sla_map


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def csch(x):
    return 2. / (np.e**x - np.e**(-x))


def sigmoid_sla_map(sla_map, weight, kernel_size=3, num_blurs=3):
    sla_map = sigmoid(weight * sla_map - 0.5 * weight)

    kernel = (kernel_size, kernel_size)
    for _ in range(num_blurs):
        sla_map_ = cv2.blur(sla_map, kernel)
        sla_map = sla_map_ * sla_map
    # sla_map = sigmoid(weight * sla_map - 0.5 * weight)

    return sla_map


def unit_vector(vector):
    '''Returns the unit vector of the vector.
    '''
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    '''Returns the angle in radians between vectors 'v1' and 'v2'
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def neigh_direction(pnt, neigh_pnt):
    '''Returns angle [rad] between two points clockwise from x-axis.

    Args:
        pnt: Coordinates of node (i,j)
        neigh_pnt: Coordinates of neighbor node (i,j)
    '''
    vec = [neigh_pnt[0] - pnt[0], neigh_pnt[1] - pnt[1]]

    # Image --> Cartesian coordinates
    vec[1] = -vec[1]

    neigh_angle = angle_between(vec, (1, 0))

    # When vector pointing downwards
    if vec[1] < 0.:
        neigh_angle = 2. * np.pi - neigh_angle

    return neigh_angle


def angle_diff(ang1, ang2):
    '''Difference in radians for two angles [rad].
    Ref: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    '''
    a = ang1 - ang2
    a = (a + np.pi) % (2. * np.pi) - np.pi
    return a


def deg2rad(deg):
    return deg * np.pi / 180.


def ang2idx(ang):
    '''
    Returns the idx corresponding to one of the angles {0, 45, 90, 125, 180,
    225, 270, 315} representing directions {L, TL, T, TR, R, BR, B, BL}.
    Args:
        ang: Radians
    '''
    for idx in range(8):
        if np.isclose(ang, deg2rad(idx * 45)):
            return idx
    raise Exception(f'Given angle {ang} ({ang*180/np.pi}) not in the set')


def compute_da_contribution_map(da_num):
    '''
    phi: Directional affordance directions
    theta: Predefined directional intervals

    Args:
        da_num: Number of discretized directions (i.e. 32)

    Returns:
        c_mat: Contribution mapping da_idx --> dir_idx

            c_mat[da_i, dir_j] --> Contrib. of p_DA(phi=i) --> p_Dir(theta=j)

                 dir_1 dir_2 ...
                ----------------
           da_1 |  1     0       <-- Sums to 1 (how da_1 is distributed among
           da_2 |  0     1           dir_1, ..., dir_N)
            ... |
    '''
    delta_phi = 2 * np.pi / da_num

    dir_num = 8

    # NOTE: Count right-most region 'R' as two regions (+45, -45)
    #       Do reduction later
    c_mat = np.zeros((da_num, dir_num + 1))

    thetas = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    thetas = [theta * np.pi / 180. for theta in thetas]
    for phi_idx in range(da_num):
        phi_0 = phi_idx * delta_phi
        phi_1 = (phi_idx + 1) * delta_phi
        for theta_idx in range(dir_num + 1):
            theta_0 = thetas[theta_idx]
            theta_1 = thetas[theta_idx + 1]

            # DA region after
            if theta_1 < phi_0:
                continue
            # DA region before
            if phi_1 < theta_0:
                break
            # Intersection 3: DA entering
            if phi_0 <= theta_0 and phi_1 <= theta_1:
                intersection = phi_1 - theta_0
            # Intersection 4: DA leaving
            elif theta_0 < phi_0 and theta_1 <= phi_1:
                intersection = theta_1 - phi_0
            # Intersection 5: DA within
            elif theta_0 <= phi_0 and phi_1 <= theta_1:
                intersection = phi_1 - phi_0
            # Intersection 6: DA enclose
            elif phi_0 <= theta_0 and theta_1 <= phi_1:
                intersection = theta_1 - phi_0
            else:
                raise Exception('Unspecified condition')

            # Contribution ratio from p(phi) --> p(theta)
            c_ratio = intersection / delta_phi

            c_mat[phi_idx, theta_idx] = c_ratio

    # Sum and reduce the first and last interval
    # (corresponding to the same -22.5 --> 22.5 interval)
    c_mat[:, 0] += c_mat[:, -1]
    c_mat = c_mat[:, :-1]

    return c_mat


def dsla_weighted_adj_mat(
        A,
        sla_map,
        da_map,
        sla_threshold=0.1,
        da_threshold=1.,
        eps=0,  #1e-12,
        smoothing_kernel_size=8,
        smoothing_power=8):
    '''
    '''
    # For penalizing paths close to border
    # sla_map = smoothen_sla_map(sla_map,
    #                            kernel_size=smoothing_kernel_size,
    #                            power=smoothing_power)
    mask = sla_map == 0
    sla_map = sigmoid_sla_map(sla_map, 6, num_blurs=3)
    sla_map[mask] = 0

    # Col count
    I, J = sla_map.shape

    # All nodes unreachable by default
    weighted_A = np.ones(A.shape) * np.inf

    # DA --> Direction contribution mapping
    # c_map[da_i, dir_j] --> Contribution of prob da_i to prob dir_j
    da_num = da_map.shape[0]
    c_map = compute_da_contribution_map(da_num)

    # Range associated with 'directionless' space (DA spread out)
    dir_prob_thresh = 0.1 * 1 / 8  # Uniform probability

    # Compute directional adjacency weight node-by-node
    # NOTE: Coordinates (i,j) == (row, col) in image coordinates
    #         (0,0) is top-left corner
    #         (127,0) is bottom-left corner
    # TODO: Get nonzero indices from SLA map
    for i in range(I):
        for j in range(J):

            # Skip nodes without SLA
            # if sla_map[i, j] < eps:
            if sla_map[j, i] <= eps:
                continue

            # Transform p(DA) --> p(Dir): p_dir[p(dir=0), ... p(dir=N)]
            # p_dir = np.zeros((8))
            # for da_idx in range(da_num):
            #     p_dir += c_map[da_idx] * da_map[
            #         da_idx, j, i]  # TODO Confirm direction (i, j)

            # p_dir_1 = [c_dir_1, c_dir_2, ... , c_dir_32] x [p_da_1, p_da_2, ... , p_da_32].T
            p_dir = np.matmul(c_map.T, da_map[:, j, i])

            # Apply convolution to allow diagonal transitions in straight
            # directional fields
            kernel = np.array([0.125, 0.75, 0.125])
            p_dir_padding = np.pad(p_dir, 1, 'wrap')
            p_dir_padding = np.convolve(p_dir_padding, kernel, mode='same')
            p_dir = p_dir_padding[1:-1]

            # Node index for current node and surrounding neighbors
            node_idx = node_coord2idx(i, j, J)
            neigh_idxs = get_neighbor_nodes(node_idx, A)

            # Compute directional adjacency neighbor-by-neighbor
            for neigh_idx in neigh_idxs:

                neigh_i, neigh_j = node_idx2coord(neigh_idx, J)

                # sla = sla_map[neigh_i, neigh_j]
                sla = sla_map[neigh_j, neigh_i]

                # Non-SLA nodes unreachable
                # if sla_map[neigh_i, neigh_j] <= eps:
                if sla_map[neigh_j, neigh_i] <= eps:
                    continue

                # Directional angle (convert to Cartesian coordinates)
                # ang = neigh_direction((j, i), (neigh_j, neigh_i))
                ang = neigh_direction((i, j), (neigh_i, neigh_j))
                dir_idx = ang2idx(ang)

                p_dir_neigh = p_dir[dir_idx]

                # If directional angle is within limits ==> Reachable node
                if p_dir_neigh > dir_prob_thresh:

                    # NEW
                    dx = neigh_i - i
                    dy = neigh_j - j
                    dist = np.sqrt((dx)**2 + (dy)**2)

                    # SLA penalty = - log( SLA )
                    # i.e. penalty incresing as SLA decreases
                    # cost = -1e6 * np.log(sla**256 + 1e-320) + 1e-3 * dist  # 1.
                    cost = csch(sla) + 1e-6 * dist

                    weighted_A[node_idx, neigh_idx] = cost

    return weighted_A


if __name__ == '__main__':

    c_mat = compute_da_contribution_map(32)

    np.set_printoptions(precision=2, suppress=True)
    print(c_mat)
    print()
