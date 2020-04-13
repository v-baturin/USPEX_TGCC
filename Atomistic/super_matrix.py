"""
USPEX.Common.Atomistic.super_matrix
===================================

Small utility to quickly generate 3d matrix series

.. codeauthor:: Maxim Rakitin
"""


def super_matrix(xmin: int, xmax: int, ymin: int, ymax: int, zmin: int, zmax: int):
    """
    This is a small utility to quickly generate a 3d matrix series of the type [x1 y1 z1; x2 y2 z2; ......]
    with x, y and z in the input ranges [x_min, x_max], [y_min, y_max] and [z_min, z_max].
    The functional is usually used to create supercells.
    Example:
        INPUT:  [0 1 0 1 0 1]
        OUTPUT: [0 0 0; 0 0 1; 0 1 0; 0 1 1; 1 0 0; 1 0 1; 1 1 0; 1 1 1]

    :type xmin: int
    :param xmin: min x value.
    :type xmax: int
    :param xmax: max x value.
    :type ymin: int
    :param ymin: min y value.
    :type ymax: int
    :param ymax: max y value.
    :type zmin: int
    :param zmin: min z value.
    :type zmax: int
    :param zmax: max z value.
    :rtype: list
    :return matrix: resulted matrix.
    """

    # matrix = []
    # for i in range(xmin, xmax + 1):
    #     for j in range(ymin, ymax + 1):
    #         for k in range(zmin, zmax + 1):
    #             matrix.append([i, j, k])
    # return matrix
    return [[i, j, k] for i in range(xmin, xmax + 1) for j in range(ymin, ymax + 1) for k in range(zmin, zmax + 1)]
