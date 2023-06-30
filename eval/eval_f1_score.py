import cv2
import numpy as np


def eval_f1_score(pred_paths: list,
                  gt_map: np.array,
                  drivable: np.array,
                  lane_width=7,
                  gt_map_dilations=3,
                  height=256,
                  width=256):
    '''
    Evaluate the F1 score over all road elements.

    Args:
        pred_paths: List of np.array (N,2) representing a trajectory of N
                    (i, j) pnts.
        gt_lane: Lanes plotted onto a dense boolean map (H, W).
        drivable: Road region plotten onto a dense boolean map (H, W).
        lane_width: 7 corresponds to 9 pixels (?).

    Returns:
        F1 score.
    '''
    ####################
    #  Prediction map
    ####################
    pred_map = np.zeros((height, width, 3), dtype=np.uint8)
    for path in pred_paths:
        pnts = path.astype(np.int32)
        pnts = pnts.reshape((-1, 1, 2))
        pred_map = cv2.polylines(pred_map, [pnts],
                                 isClosed=False,
                                 color=(255, 255, 255),
                                 thickness=lane_width)
    pred_map = pred_map / 255
    pred_map = pred_map[:, :, 0]
    pred_map = pred_map.astype(bool)

    ############
    #  GT map
    ############
    gt_map = (255. * gt_map).astype(np.uint8)
    gt_map = np.expand_dims(gt_map, -1)
    gt_map = np.tile(gt_map, (1, 1, 3))
    kernel = np.ones((3, 3), np.uint8)
    gt_map = cv2.dilate(gt_map, kernel, iterations=gt_map_dilations)
    gt_map = gt_map / 255
    gt_map = gt_map[:, :, 0]
    gt_map = gt_map.astype(bool)

    tp = np.logical_and(pred_map, gt_map) + np.logical_and(~pred_map, ~gt_map)
    fp = pred_map * np.logical_xor(pred_map, tp)
    fn = gt_map * np.logical_xor(gt_map, tp)

    tp[drivable == 0] = 0
    fp[drivable == 0] = 0
    fn[drivable == 0] = 0

    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)
    fn_sum = np.sum(fn)
    f1_score = 2 * tp_sum / (2 * tp_sum + fp_sum + fn_sum)

    return f1_score
