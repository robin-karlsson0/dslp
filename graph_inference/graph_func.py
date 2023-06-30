import copy

import cv2
import matplotlib.pyplot as plt
import numpy as np

#from graph_inference.a_star import a_star
from graph_inference.dense_nms import dense_nonmax_sup
from graph_inference.dsla_weight_matrix import (angle_between,
                                                dsla_weighted_adj_mat)
from graph_inference.grid_map import (get_neighbor_nodes, grid_adj_mat,
                                      node_coord2idx, node_idx2coord)
from losses.da_model_free_kl_div import integrate_distribution

# from preproc.conditional_dsla import comp_descrete_entry_points


def discretize_border_regions(dense_map, value, nms_pnts=[]):
    '''
    Ref: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
    '''
    # Convert to binary map
    dense_map = dense_map == value
    dense_map = (255. * dense_map).astype(np.uint8)
    # NOTE: Dilate in order to reduce posibility of zero-thickness clusters
    # NOTE: May remove overlap with existing NMS points
    kernel = np.ones((3, 3), np.uint8)
    dense_map = cv2.dilate(dense_map, kernel, iterations=1)
    dense_map = cv2.erode(dense_map, kernel, iterations=2)
    dense_map = cv2.dilate(dense_map, kernel, iterations=1)  # NOTE 2 !!!

    # Find separated clusters
    # _, contours, _ = cv2.findContours(dense_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(dense_map, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)

    entry_pnt_list = []
    for c in contours:

        # Skip if a NMS point already exist within contour
        is_inside = -1
        for nms_pnt in nms_pnts:
            nms_pnt_rev = (nms_pnt[1], nms_pnt[0])
            is_inside = cv2.pointPolygonTest(c, nms_pnt_rev, False)
            if is_inside == 1 or is_inside == 0:
                break

        if is_inside == 1 or is_inside == 0:
            continue

        # Calculate moments for each contour
        M = cv2.moments(c)
        # Skip bad shapes
        if M["m10"] == 0 or M["m00"] == 0 or M["m01"] == 0:
            continue
        # Calculate x,y coordinate of center
        c_x = int(M["m10"] / M["m00"])
        c_y = int(M["m01"] / M["m00"])

        entry_pnt_list.append((c_y, c_x))

    return entry_pnt_list


# def search_path(A,
#                 weighted_A,
#                 start_node_idx,
#                 goal_node_idx,
#                 I,
#                 J,
#                 heuristic_type="max_axis"):
#     '''
#     '''
# 
#     d, par = a_star(A, weighted_A, start_node_idx, goal_node_idx, J,
#                     "manhattan")
# 
#     d_arr = np.array(d).reshape((I, J))
#     d_arr[d_arr == np.inf] = 0.
# 
#     path = []
# 
#     par_idx = goal_node_idx
#     while True:
# 
#         path.insert(0, par_idx)
#         par_idx = par[par_idx]
# 
#         # Goal is unreachable
#         if par_idx == -1:
#             break
# 
#     return d_arr, path


def path_idx2coord(path, J):
    '''Converts a list of vertex indices [idx_1, ] to coordinates [[i,j]_1, ]
    '''
    coords = []
    for path_vert in path:
        i = (int(path_vert / J))
        j = (path_vert % J)
        coords.append([i, j])

    return coords


# def compute_path_divergence(start_pnt, pnts):
#     '''Returns the divergence [rad] between a path and a set of paths. Each path
#     is represented by points.
# 
#     Divergence means the angle spanned by the path direction relative to single
#     point. The path direction is represented by a path coordinate ahead of the
#     starting point.
# 
#     Example:
#         start_pnt = path_1[i]
#         pnts = [ path_1[i+5], path_2[i+5], path_3[i+5] ]
# 
#     Args:
#         start_pnt (tuple): Single point representing path center (i,j).
#         pnts (list): List of tuples representing future direction of paths.
# 
#     Returns:
#         (float): Divergence angle [rad].
#     '''
#     # Convert points to vectors
#     vecs = []
#     for pnt in pnts:
#         dx = pnt[0] - start_pnt[0]
#         dy = pnt[1] - start_pnt[1]
#         vec = np.array([dx, dy])
#         vecs.append(vec)
# 
#     # Order points in counter-clockwise order
#     angs = []
#     for vec in vecs:
#         ang = angle_between(vec, [1, 0])
#         angs.append(ang)
# 
#     ordered_angs_idxs = np.argsort(angs)
# 
#     delta_angs = []
#     # Angle between vectors
#     for idx in range(len(ordered_angs_idxs) - 1):
#         vec1_idx = ordered_angs_idxs[idx]
#         vec2_idx = ordered_angs_idxs[idx + 1]
# 
#         vec1 = vecs[vec1_idx]
#         vec2 = vecs[vec2_idx]
# 
#         delta_ang = angle_between(vec1, vec2)
#         delta_angs.append(delta_ang)
# 
#     # Angle between last and first vector
#     delta_ang = 2. * np.pi - sum(delta_angs)
#     delta_angs.append(delta_ang)
# 
#     div_ang = np.sum(delta_angs) - np.max(delta_angs)
# 
#     return div_ang


# def find_fork_point(path_list, div_ang_threshold, lookahead_idx):
#     '''Finds the point where a set of paths diverges.
# 
#     NOTE: The earliest fork point is the second point.
# 
#     Args:
#         path_list (list): List of lists of point coordinate tuples.
#         div_ang_threshold (float): Exceeding this angle [rad] denotes
#                                     diverging paths.
# 
#     Returns:
#         (int) List index, or 'None' for single and non-diverging paths.
#     '''
#     N = np.min([len(path) for path in path_list])
#     forking_pnt = None
#     for i in range(1, N - lookahead_idx):
#         start_pnt = path_list[0][i]
# 
#         pnts = [pnt[i + lookahead_idx] for pnt in path_list]
# 
#         div_ang = compute_path_divergence(start_pnt, pnts)
# 
#         if div_ang > np.pi / 4:
#             #forking_pnt = i PREV
#             break
# 
#         forking_pnt = i
# 
#     return forking_pnt


# def unify_entry_paths(path_list, div_ang_threshold, lookahead_idx):
#     '''Unifies all path coordinates up to the fork point.
# 
#     Args:
#         path_list (list): List of lists of point coordinate tuples.
#         div_ang_threshold (float): Exceeding this angle [rad] denotes
#                                     diverging paths.
#     '''
#     if len(path_list) == 1:
#         start_pnt = path_list[0][0]
#         end_pnt = path_list[0][1]
#         entry_path = [start_pnt, end_pnt]
# 
#         connecting_paths = [path[1:] for path in path_list]
# 
#         return entry_path, connecting_paths
# 
#     # Find path until all paths start to diverge
#     fork_pnt = find_fork_point(path_list, div_ang_threshold, lookahead_idx)
# 
#     start_pnt = path_list[0][0]
#     if fork_pnt:
#         end_pnt = path_list[0][fork_pnt]
#     else:
#         end_pnt = path_list[0][1]
# 
#     entry_path = [start_pnt, end_pnt]
#     # Replace the entry path with the common path
#     connecting_paths = [path[fork_pnt:] for path in path_list]
# 
#     return entry_path, connecting_paths


def comp_entry_exit_pnts(out_sla,
                         out_da,
                         lane_width_px,
                         region_width=2,
                         p_tresh=0.4):
    '''
    Coordinate system:
            i
          -------------------
       j |  0   1  ...  128  |
         |  1                |
         | ...               |
         | 128               |
          -------------------

    Args:
        out_sla:
        out_da: (C,H,W) Categorical probability distribution of directionality

    Returns:
        List of entry points [(i,j), ... ] and exit points [(i,j), ... ]
    '''
    # Points from NMS
    mask = np.zeros_like(out_sla, dtype=bool)
    mask[region_width:-region_width, region_width:-region_width] = True
    out_sla_nms = copy.deepcopy(out_sla)
    out_sla_nms[mask] = 0
    out_sla_nms = dense_nonmax_sup(out_sla_nms, lane_width_px)

    # Compute element in none (0) | in (1) | out (2) state
    num_ang_discr = out_da.shape[0]
    idx_0_deg = 0
    idx_90_deg = num_ang_discr // 4
    idx_180_deg = num_ang_discr // 2
    idx_270_deg = (3 * num_ang_discr) // 4

    flow_entry = np.zeros((128, 128), dtype=int)
    flow_exit = np.zeros_like(flow_entry)

    H, W = out_sla.shape
    # Top side (i: 0, j: 0 --> J)
    for i in range(0, region_width):
        for j in range(0, W):
            if out_sla[i, j] == 0:
                continue
            p_exit = np.sum(out_da[idx_0_deg:idx_180_deg, i, j])
            p_entry = 1 - p_exit
            if p_exit > p_tresh:
                flow_exit[i, j] = 1
            if p_entry > p_tresh:
                flow_entry[i, j] = 1

    # Right side (i: 0 --> I, j: J )
    for i in range(0, W):
        for j in range(H - region_width, H):
            if out_sla[i, j] == 0:
                continue
            p_entry = np.sum(out_da[idx_90_deg:idx_270_deg, i, j])
            p_exit = 1 - p_entry
            if p_exit > p_tresh:
                flow_exit[i, j] = 1
            if p_entry > p_tresh:
                flow_entry[i, j] = 1

    # Bottom side (i: I, j: 0 --> J)
    for i in range(W - region_width, W):
        for j in range(0, H):
            if out_sla[i, j] == 0:
                continue
            p_entry = np.sum(out_da[idx_0_deg:idx_180_deg, i, j])
            p_exit = 1 - p_entry
            if p_exit > p_tresh:
                flow_exit[i, j] = 1
            if p_entry > p_tresh:
                flow_entry[i, j] = 1

    # Left side (i: 0 --> 128, j: 0)
    for i in range(0, W):
        for j in range(0, region_width):
            if out_sla[i, j] == 0:
                continue
            p_exit = np.sum(out_da[idx_90_deg:idx_270_deg, i, j])
            p_entry = 1 - p_exit
            if p_exit > p_tresh:
                flow_exit[i, j] = 1
            if p_entry > p_tresh:
                flow_entry[i, j] = 1

    # Points from NMS
    i_idxs, j_idxs = np.where(out_sla_nms > 0.05)
    nms_pnts = [pnt for pnt in zip(i_idxs.tolist(), j_idxs.tolist())]

    # Point direction
    entry_pnts = []
    exit_pnts = []
    for nms_pnt in nms_pnts:
        i, j = nms_pnt
        if flow_entry[i, j] == 1:
            entry_pnts.append(nms_pnt)
        if flow_exit[i, j] == 1:
            exit_pnts.append(nms_pnt)
        # else:
        #     entry_pnts.append(nms_pnt)
        #     exit_pnts.append(nms_pnt)

    # Points from grouping
    entry_pnts += discretize_border_regions(flow_entry, 1, entry_pnts)
    exit_pnts += discretize_border_regions(flow_exit, 1, exit_pnts)

    # Remove points too far from any SLA prediction
    in_sla_region = out_sla > 0.05
    # out_sla_grown = (255 * out_sla_grown).astype(np.uint8)
    # kernel = np.ones((3, 3), np.uint8)
    # out_sla_grown = cv2.dilate(out_sla_grown, kernel, iterations=5)
    # out_sla_grown /= 255
    entry_pnts_ = []
    for pnt in entry_pnts:
        i = pnt[0]
        j = pnt[1]
        if in_sla_region[i, j]:
            entry_pnts_.append(pnt)
    entry_pnts = entry_pnts_

    exit_pnts_ = []
    for pnt in exit_pnts:
        i = pnt[0]
        j = pnt[1]
        if in_sla_region[i, j]:
            exit_pnts_.append(pnt)
    exit_pnts = exit_pnts_

    return entry_pnts, exit_pnts


def viz_entry_exit_pnts(sla_map, entry_pnts, exit_pnts):
    for pnt in entry_pnts:
        i = pnt[0]
        j = pnt[1]
        sla_map[i, j] = -1
    for pnt in exit_pnts:
        i = pnt[0]
        j = pnt[1]
        sla_map[i, j] = 2
    plt.imshow(sla_map)
    plt.show()


def viz_weighted_A(sla_map, A, weighted_A, eps=0, scale_factor=10, t=1, l=0.1):

    dim = (128 * scale_factor, 128 * scale_factor)
    sla_map_viz = cv2.resize(sla_map, dim, interpolation=cv2.INTER_LINEAR)
    sla_map_viz = (255 * sla_map_viz).astype(np.uint8)
    sla_map_viz = cv2.applyColorMap(sla_map_viz, cv2.COLORMAP_HOT)
    sla_map_viz = cv2.cvtColor(sla_map_viz, cv2.COLOR_BGR2RGB)

    I, J = sla_map.shape
    for i in range(0, I, 3):
        for j in range(0, J, 3):

            # Skip nodes without SLA
            # if sla_map[i, j] < eps:
            if sla_map[j, i] <= eps:
                continue

            # Node index for current node and surrounding neighbors
            node_idx = node_coord2idx(i, j, J)
            neigh_idxs = get_neighbor_nodes(node_idx, A)

            # Compute directional adjacency neighbor-by-neighbor
            for neigh_idx in neigh_idxs:

                if weighted_A[node_idx, neigh_idx] < np.inf:

                    neigh_i, neigh_j = node_idx2coord(neigh_idx, J)

                    pnt0 = (i * scale_factor, j * scale_factor)
                    pnt1 = (neigh_i * scale_factor, neigh_j * scale_factor)
                    sla_map_viz = cv2.arrowedLine(sla_map_viz,
                                                  pnt0,
                                                  pnt1, (0, 0, 255),
                                                  thickness=t,
                                                  tipLength=l)

    plt.imshow(sla_map_viz)
    plt.show()


def comp_entry_exits(out_sla, out_da, lane_width_px=6):
    '''
    Returns:
        entry_paths (list):     [ [(i,j)_0, None], ... ]
        connecting_pnts (list): [ None ]
        exit_paths (list):      [ [None, (i.j)_1], ... ]
    '''
    entry_pnts, exit_pnts = comp_entry_exit_pnts(out_sla, out_da,
                                                 lane_width_px)

    # Build DAG
    entry_paths = [[(pnt[1], pnt[0]), None] for pnt in entry_pnts]
    exit_paths = [[None, (pnt[1], pnt[0])] for pnt in exit_pnts]

    connecting_paths = [None]

    return entry_paths, connecting_paths, exit_paths


def comp_graph(out_sla,
               out_da,
               lane_width_px=6,
               div_ang_threshold=np.pi / 8,
               lookahead_idx=6,
               scale=1.):
    '''
    Args:
        out_sla:   (128,128) Numpy matrices
        out_entry:
        out_exit:
        out_dir_0:
        out_dir_1:
        out_dir_2:
        div_ang_threshold:
        lookahead_idx:

    Returns:
        entry_paths (list):     [ [(i,j)_0, (i.j)_1], ... ]
        connecting_pnts (list): [ [(i,j)_0, (i.j)_1], ... ]
        exit_paths (list):      [ [(i,j)_0, (i.j)_1], ... ]
    '''
    ###################
    #  Preprocessing
    ###################
    # Smoothen SLA field
    # Determine entry and exit points
    entry_pnts, exit_pnts = comp_entry_exit_pnts(out_sla, out_da,
                                                 lane_width_px)

    # List with (i,j) coordinates as tuples
    # NOTE: Origo is bottom left when viewed as plot
    #       ==> Need to switch coordinates for 'entry' and 'exit' points
    # entry_pnts = comp_descrete_entry_points(out_entry, scale)
    # exit_pnts = comp_descrete_entry_points(out_exit, scale)

    # Eight-directional connected grid world adjacency matrix
    I, J = (128, 128)
    A = grid_adj_mat(I, J, "8")

    # out_da = np.roll(out_da, 8, axis=0)

    weighted_A = dsla_weighted_adj_mat(A, out_sla, out_da)

    # out_da_perm = np.zeros_like(out_da)
    # out_da_perm[]

    # out_da2 = np.zeros_like(out_da)
    # out_da2[26] = 1
    # weighted_A = dsla_weighted_adj_mat(A, out_sla, out_da2)

    # viz_weighted_A(out_sla, A, weighted_A)

    ###
    entry_paths = []
    connecting_paths = []
    exit_paths = []
    ###

    tree_list = []
    ###

    for entry_pnt in entry_pnts:
        # for entry_pnt in [(126, 8)]:
        # for entry_pnt in [(0, 20), (22, 125, (127, 73))]:
        print(f"Entry point: {entry_pnt}")

        # NOTE: Need to switch coordinates
        start_i = entry_pnt[1]
        start_j = entry_pnt[0]
        start_node_idx = node_coord2idx(start_i, start_j, J)

        path_list = []

        for exit_pnt in exit_pnts:
            # for exit_pnt in [(107, 1)]:
            print(f"    Search for exit point: {exit_pnt}")

            goal_i = exit_pnt[1]
            goal_j = exit_pnt[0]
            goal_node_idx = node_coord2idx(goal_i, goal_j, J)

            d_arr, path = search_path(A, weighted_A, start_node_idx,
                                      goal_node_idx, I, J)

            # Skip unreachable goal
            if len(path) == 1:
                continue

            path = path_idx2coord(path, J)

            path_list.append(path)

        # Skip entry point not connected to any exit point
        if len(path_list) == 0:
            continue

        # NOTE: SHOULD BE DONE WHILE CHECKING END POINTS TOO
        #       (OTHERWISE REDUCE TO EARLY AND NOT CONNECT)
        entry_path, connecting_paths = unify_entry_paths(
            path_list, div_ang_threshold, lookahead_idx)
        entry_paths.append(entry_path)
        if connecting_paths:
            #    connecting_paths += connecting_paths_
            tree_list.append(connecting_paths)

    connecting_paths = []

    # Unify exit paths in all trees
    for exit_pnt in exit_pnts:

        # Reverse (i,j) coordinates
        exit_pnt = exit_pnt[::-1]

        # For each exit point, find all paths in all trees
        # Each tree can only have one such path
        exit_path_dicts = []
        for tree_idx in range(len(tree_list)):

            path_list = tree_list[tree_idx]

            for path_idx in range(len(path_list)):

                path = path_list[path_idx]

                #print(exit_pnt, tuple(path[-1]), tuple(path[-1]) == exit_pnt)
                if tuple(path[-1]) == exit_pnt:
                    match_dict = {'tree_idx': tree_idx, 'path_idx': path_idx}
                    exit_path_dicts.append(match_dict)

        # Collect paths into a path_list
        # Reverse paths
        # Unify paths
        # Reverse paths
        # Replace paths

        path_list = []
        for dict_idx in range(len(exit_path_dicts)):
            tree_idx = exit_path_dicts[dict_idx]['tree_idx']
            path_idx = exit_path_dicts[dict_idx]['path_idx']
            path = tree_list[tree_idx][path_idx]
            path_list.append(path)

        if len(path_list) == 0:
            continue

        # Reverse all paths
        path_list = [path[::-1] for path in path_list]

        exit_path, connecting_paths_ = unify_entry_paths(
            path_list, div_ang_threshold, lookahead_idx)

        # Reverse all paths
        exit_path = exit_path[::-1]
        connecting_paths_ = [path[::-1] for path in connecting_paths_]

        exit_paths.append(exit_path)
        connecting_paths += connecting_paths_

    # Build DAG
    entry_pnts = [path[0] for path in entry_paths]
    fork_pnts = [path[1] for path in entry_paths]
    join_pnts = [path[0] for path in exit_paths]
    exit_pnts = [path[1] for path in exit_paths]

    connecting_pnts = [[path[0], path[-1]] for path in connecting_paths]

    return entry_paths, connecting_pnts, exit_paths
