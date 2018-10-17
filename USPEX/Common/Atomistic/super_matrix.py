__author__ = 'mrakitin'

# import numpy as np

def super_matrix(xmin, xmax, ymin, ymax, zmin, zmax):
    """
    This is a small utility to quickly generate a 3d matrix series such as [1 0 0; 0 1 0; 0 0 1; ........].
    The functional is usually used to create super cell.
    Example:
        INPUT:  [0 1 0 1 0 1]
        OUTPUT: [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]

    :param xmin: min x value.
    :param xmax: max x value.
    :param ymin: min y value.
    :param ymax: max y value.
    :param zmin: min z value.
    :param zmax: max z value.
    :return matrix: resulted matrix.
    """

    # matrix = []
    # for i in range(xmin, xmax + 1):
    #     for j in range(ymin, ymax + 1):
    #         for k in range(zmin, zmax + 1):
    #             matrix.append([i, j, k])
    # return matrix
    return [[i,j,k] for i in range(xmin, xmax + 1) for j in range(ymin, ymax + 1) for k in range(zmin, zmax + 1)]


if __name__ == "__main__":
    matrix = super_matrix(-2, 2, -2, 2, -2, 2)
    print(matrix)
