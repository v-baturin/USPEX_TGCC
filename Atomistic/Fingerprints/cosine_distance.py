"""
USPEX.Common.Atomistic.Fingerprints.cosine_distance
===================================================

Calculation of cosine distances

.. codeauthor:: Maxim Rakitin
"""

import numpy as np


def cosine_distance(matrA, matrB, weight):
    """
    Calculation of cosine distances using eq.(6b) from JCP-2009.

    :type matrA: numpy array
    :param matrA: atomic fingerprint 1.
    :type matrB: numpy array
    :param matrB: atomic fingerprint 2.
    :type weight: list or int
    :param weight: weight for a particular atom type in the cell.
    :rtype: float
    :return: resulted cosine distance.
    """
    if type(weight) is list and len(weight) == 1:
        weight = weight[0]
    matrA = np.asarray(matrA)
    matrB = np.asarray(matrB)
    if type(weight) is int and weight != 1:
        try:
            weight = np.diag(weight)
        except ValueError:
            weight = np.diag([weight])

    coef1 = np.sum(np.dot(weight, matrA * matrB))
    coef2 = np.sum(np.dot(weight, matrA * matrA))
    coef3 = np.sum(np.dot(weight, matrB * matrB))

    dist = (1 - coef1 / (coef2 * coef3) ** 0.5) / 2

    return dist
