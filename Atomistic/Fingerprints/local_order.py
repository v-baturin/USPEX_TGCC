__author__ = 'mrakitin'

import numpy as np


def local_order(V, numIons, atom_fing, delta):
    '''
    
    :param V: 
    :param numIons: 
    :param atom_fing: 
    :param delta: 
    :return: 
    '''

    N_atom = sum(numIons)
    if N_atom == 0:
        return np.nan
    N_type = numIons.shape[0]
    order = np.zeros(N_atom)
    weight = numIons[:] / float(N_atom)  # weight factor
    for i in range(N_atom):
        for j in range(N_type):
            order[i] += weight[j] * sum(atom_fing[i, j, :] ** 2)  # eq. 5 in CPC-2010
    order *= delta / (float(V) / float(N_atom)) ** (1.0 / 3.0)
    order **= 0.5
    return order
