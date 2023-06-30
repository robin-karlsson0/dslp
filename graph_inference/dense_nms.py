import numpy as np


def dense_nonmax_sup(array, m, threshold=0.0):
    '''Reduces a dense array to its maximum values (or "peaks").
    The algorithm iterates through each array element (i, j) one by one, and
    finds the highest value in its neighborhood (m, m). If (i, j) IS NOT the
    highest value, (i, j) is suppressed to zero.
    Args:
        array: Dense 2D float array of shape (n, n)
        m (int): Neighborhood size
        threshold (float): Elements bellow value zero regardles of neighborhood
    Returns:
        array_sup: Dense 2D float array with only maximum valued elements
                   remain non-zero.
    '''
    # Add padding to enable regular iteration
    array_pad = np.pad(array, m, "constant")
    array_sup = np.copy(array_pad)

    for i in range(m, array_pad.shape[0] - m):
        for j in range(m, array_pad.shape[1] - m):
            # Slice (m, m) neighbourhood around (i, j)
            neigh_array = array_pad[i - m:i + m, j - m:j + m]
            # Get maximum value in neighbourhood
            neigh_max = np.max(neigh_array)
            # Set (i, j) to zero if not largets value
            if (array_pad[i, j] < neigh_max or array_pad[i, j] < threshold):
                array_sup[i, j] = 0.0

    # Remove padding
    array_sup = array_sup[m:-m, m:-m]

    return array_sup
