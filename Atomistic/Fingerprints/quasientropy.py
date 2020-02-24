from __future__ import division

__author__ = 'mrakitin'

import numpy as np

from .cosine_distance import cosine_distance


def quasientropy(structure, atom_fing):
    """
    Calculate structure quasientropy.
    :param numIons: number of atoms.
    :param atom_fing: atomic fingerprints.
    :return sQE: structure quasientropy.
    """

    # assert check1DArray(numIons, int)
    numIons = np.unique(structure.get_chemical_symbols(),return_counts=True)[1]
    sQE = 0.0
    weight = numIons / np.sum(numIons)

    for i in range(numIons.shape[0]):
        if numIons[i] > 1:
            tmp = 0
            k = 0
            start_n = np.sum(numIons[0:i])
            for j in range(start_n, start_n + numIons[i]):
                tmp_fing1 = atom_fing[j, :, :]
                for j1 in range(j + 1, start_n + numIons[i]):
                    k += 1
                    tmp_fing2 = atom_fing[j1, :, :]

                    dist = cosine_distance(tmp_fing1, tmp_fing2, weight)

                    '''
                    if abs(dist - 1.0) < 0.000001:
                        dist = 0.99999
                    '''

                    tmp += (1 - dist) * np.log(1 - dist)

            sQE += weight[i] * tmp / k

    return -sQE


if __name__ == "__main__":
    from read_poscar import read_poscar

    lat, atomType, numIons, coor = read_poscar('POSCAR_8')

    from make_matrices import make_matrices

    V, dist_matrix = make_matrices(lat, coor, numIons)

    from fingerprint import fingerprint

    order, fing, atom_fing = fingerprint(V, dist_matrix, numIons)

    sQE = quasientropy(numIons, atom_fing)
