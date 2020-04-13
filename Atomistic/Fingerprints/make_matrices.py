"""
USPEX.Common.Atomistic.Fingerprints.make_matrices
=================================================

Functions used for fingerprint calculation

.. codeauthor:: Maxim Rakitin
"""

import numpy as np
from scipy.spatial.distance import cdist

from ..super_matrix import super_matrix


def fp_weight(structure):
    """
    The function calculates weight used in fingerprint calculations.

    :type structure: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
    :param structure: system for which we want to calculate the weight.
    :rtype: numpy array
    :return: calculated weight.
    """

    # assert check1DArray(numIons, int)

    numIons = np.unique(structure.get_chemical_symbols(), return_counts=True)[1]

    L = numIons.shape[0]
    S = 0
    weight = np.zeros((L * L), dtype=float)
    for i in range(L):
        for j in range(L):
            weight[i * L + j] = numIons[i] * numIons[j]
            S += numIons[i] * numIons[j]

    weight /= S

    return weight


def make_matrices(structure, Rmax=10.0):
    """
    The function prepares matrices for fingerprint calculation.

    :type structure: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
    :param structure: system for which we want to calculate the distance matrix.
    :type Rmax: float
    :param Rmax: distance cutoff.
    :rtype: numpy array
    :return: distance matrix of the form [atom_i, atom_j, atomi_type, atomj_type, dist].
    """

    assert isinstance(Rmax, float) and Rmax >= 0

    coor = structure.get_scaled_positions()
    lat = structure.get_cell()
    numIons = np.unique(structure.get_chemical_symbols(), return_counts=True)[1]
    coor = coor[np.argsort(structure.get_chemical_symbols())]

    coor = coor - np.floor(coor)  # scale it the [0 1]
    N_atom = np.sum(numIons)

    types = []
    for i in range(len(numIons)):
        types += [i] * numIons[i]
    types = np.asarray(types)

    dist_matrix = np.zeros((0, 4), dtype=float)

    '''
    Search for all atomic pairs:
    Here we count the coordination of each atom one by one.
    Firstly we construct the possible direction matrix such as [1 0 0; 0 1 0; 0 0 1; ...
    Ideally, the best way to deal with the irregular shape is to do lattice optimization.
    But we don't consider it for now.
    '''

    # This is a direction matrix, maximally consider the direction of [1 2 0]:
    matrix = np.asarray(super_matrix(-2, 2, -2, 2, -2, 2))

    # The idea is to find all zero values [0, 0, 0] and to remove them to avoid division by zero later.
    to_delete = np.all(matrix == 0, axis=1)
    matrix = np.delete(matrix, np.where(to_delete), axis=0)

    for i in range(N_atom):
        target = np.zeros(matrix.shape)

        # Build the super cell matrix:
        for j in range(matrix.shape[0]):
            target[j] = Rmax / np.linalg.norm(np.dot(matrix[j, :], lat)) * matrix[j, :] + coor[i, :]

        lenX1 = int(np.floor(min(target[:, 0])))
        lenY1 = int(np.floor(min(target[:, 1])))
        lenZ1 = int(np.floor(min(target[:, 2])))
        lenX2 = int(np.floor(max(target[:, 0])))
        lenY2 = int(np.floor(max(target[:, 1])))
        lenZ2 = int(np.floor(max(target[:, 2])))

        matrix_tmp = np.asarray(super_matrix(lenX1, lenX2, lenY1, lenY2, lenZ1, lenZ2))

        # Obtain the distances by vectorization, pdist is used here:
        N_matrix = matrix_tmp.shape[0]
        S_coor = np.tile(coor, (N_matrix, 1))

        # S_Matrix  = reshape(repmat(Matrix_tmp, 1, N_atom)', 3, N_atom*N_Matrix)';

        S_matrix = np.tile(matrix_tmp, (1, N_atom))
        S_matrix = np.transpose(S_matrix)
        S_matrix = np.reshape(S_matrix, (3, N_atom * N_matrix), 'F')
        S_matrix = np.transpose(S_matrix)
        # For debug:
        # for j in range(S_matrix.shape[0]):
        #    print '%i ' * 3 % tuple(S_matrix[j])

        tmp_dist = cdist(np.reshape(np.dot(coor[i], lat), (1, 3)), np.dot(S_coor + S_matrix, lat))
        # For debug:
        # for j in range(tmp_dist.shape[1]):
        #    print '%.12f' % (float(tmp_dist[0][j]))

        tmp_type = np.tile(types, N_matrix)
        # For debug:
        # for j in range(tmp_type.shape[0]):
        #    print tmp_type[j]


        to_delete = np.where((tmp_dist > Rmax) | (tmp_dist < 0.5))
        # For debug:
        # for j in range(to_delete[1].shape[0]):
        #    print to_delete[1][j]

        tmp_dist = np.delete(tmp_dist, to_delete[1], axis=1)  # reset all the distances
        # For debug:
        # for j in range(tmp_dist.shape[1]):
        #    print '%.12f' % (float(tmp_dist[0][j]))

        tmp_type = np.delete(tmp_type, to_delete[1], axis=0)
        # For debug:
        # for j in range(tmp_type.shape[0]):
        #    print tmp_type[j]

        if tmp_dist.shape[1] > 0:  # sometimes you can meet an isolated atom
            tmp2 = np.zeros((tmp_dist.shape[1], 4))
            tmp2[:, 0] = i
            tmp2[:, 1] = types[i]
            tmp2[:, 2] = tmp_type
            tmp2[:, 3] = tmp_dist
            dist_matrix = np.vstack((dist_matrix, tmp2))

    return dist_matrix
