__author__ = 'mrakitin'

import numpy as np


def structure_order(f, volume, structure, weight, deltaFing=0.08):
    """
    Calculate structure order.
    :param f: fingerprint values.
    :param volume: volume of the cell.
    :param numIons: number of atoms.
    :param deltaFing: parameter delta.
    :param weight: weight.
    :return s_order: structure order.
    """

    '''
    s_order = 0.0

    r, c = f.shape
    for j in range(r):
        order_ab = 0.0
        for k in range(c):
            order_ab += f[j, k] ** 2.0

        order_ab *= deltaFing / (volume / float(sum(numIons))) ** (1.0 / 3.0)
        s_order += order_ab * weight[j]
    '''
    numIons = np.unique(structure.get_chemical_symbols(), return_counts=True)[1]
    s_order = np.sum(np.dot(weight, f**2))
    s_order *= deltaFing / (volume / float(sum(numIons))) ** (1.0 / 3.0)
    s_order **= 0.5

    return s_order


if __name__ == "__main__":
    from read_poscar import read_poscar

    lat, atomType, numIons, coor = read_poscar('POSCAR_8')

    from make_matrices import make_matrices, fp_weight

    V, dist_matrix = make_matrices(lat, coor, numIons)

    from fingerprint import fingerprint

    order, fing, atom_fing = fingerprint(V, dist_matrix, numIons)

    N_size = 1
    numIons = N_size * np.asarray(numIons)
    weight = fp_weight(numIons)

    s_order = structure_order(fing, V, numIons, weight, 0.08)
    print('Structure order:', s_order)