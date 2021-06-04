import numpy as np

from .latConverter import latConverter


def unitCellFromPrimitive(cellType, lat_prim, coor_prim):
    """
    The function converts a primitive unit cell to a conventional unit cell.
    :param cellType: cell type (A, B, C, F, I, R).
    :param lat_prim: primitive lattice.
    :param coor_prim: primitive coordinates.
    :return lat_conv: conventional lattice.
    :return coor_conv: conventional coordinates.
    """

    # Make sure we operate with NumPy arrays:
    lat_prim = np.asarray(lat_prim)
    coor_prim = np.asarray(coor_prim)

    if cellType == 'F':  # face centered
        U = [
            [-1, +1, +1],
            [+1, -1, +1],
            [+1, +1, -1],
        ]
        Duplicate = [
            [0, 0, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0],
        ]

    elif cellType == 'I':  # body centered
        U = [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
        Duplicate = [
            [0, 0, 0],
            [0.5, 0.5, 0.5],
        ]

    elif cellType == 'C':
        U = [
            [1, 1, 0],
            [1, -1, 0],
            [0, 0, 1],
        ]
        Duplicate = [
            [0, 0, 0],
            [0.5, 0.5, 0],
        ]

    elif cellType == 'B':  # group numbers 38-41
        U = [
            [0, 0, 1],
            [1, -1, 0],
            [1, 1, 0],
        ]
        Duplicate = [
            [0, 0, 0],
            [0, 0.5, 0.5],
        ]

    elif cellType == 'A':
        U = [
            [0, 1, -1],
            [0, 1, 1],
            [1, 0, 0],
        ]
        Duplicate = [
            [0, 0, 0],
            [0.5, 0.5, 0],
        ]

    elif cellType == 'R':  # rhombohedral
        U = [
            [1, -1, 0],
            [0, 1, -1],
            [1, 1, 1],
        ]
        Duplicate = [
            [0, 0, 0],
            [-1 / 3.0, 1 / 3.0, 1 / 3.0],
            [-2 / 3.0, 2 / 3.0, 2 / 3.0],
        ]

    U = np.asarray(U, dtype=float)
    Duplicate = np.asarray(Duplicate, dtype=float)

    N_atom = coor_prim.shape[0]
    N_Dupl = Duplicate.shape[0]

    lat_conv = np.dot(U, lat_prim)  # lattice in general (non triangular) representation

    # Fractional coordinates in lat_conventional:
    coor_prim = np.dot(np.dot(coor_prim, lat_prim), np.linalg.inv(lat_conv))

    coor_conv = np.zeros((N_atom * N_Dupl, 3))

    # We must add the atoms one by one (e.g., 4Cu+4O -> 8Cu+8O):
    for i in range(N_atom):
        for j in range(N_Dupl):
            coor_conv[N_Dupl * i + j, :] = coor_prim[i, :] + Duplicate[j, :]

    coor_conv -= np.floor(coor_conv)

    lat_conv = latConverter(latConverter(lat_conv))  # back to real lattice in triangular form

    return lat_conv, coor_conv
