"""
USPEX.Common.Atomistic.Fingerprints.local_order
===============================================

Local order calculation function

.. codeauthor:: Maxim Rakitin
"""

import numpy as np


def local_order(V, numIons, atom_fing, delta):
    """
    Local order calculation function.

    :type V: float
    :param V: volume of the cell.
    :type numIons: numpy array
    :param numIons: vector of integers expressing the abundance of each atomic type (or molecule) in the unit cell.
    :type atom_fing: numpy array
    :param atom_fing: tomic fingerprint.
    :type delta: float
    :param delta: bin width.
    :rtype: numpy array
    :return: local order.
    """
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
