"""
USPEX.Common.Atomistic.optLattice
=================================

Optimizes a too elongated lattice

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from __future__ import division

import numpy as np


# TODO check this functions and write tests for them

def reoptimizeVector(v1: np.ndarray, v2: np.ndarray, flag1: int):
    """
    The function reoptimizes a vector against another vector. Needs better description.

    :type v1: numpy array
    :param v1: vector to reoptimize.
    :type v2: numpy array
    :param v2: vector against which we reoptimize v1.
    :type flag1: int
    :param flag1: flag that is raised if a new v1 has been found with norm less than the original v1.
    :rtype: tuple
    :return: (v1, flag)
    """

    flag = flag1

    # Convert vectors to ndarray:
    v1 = np.array(v1)
    v2 = np.array(v2)

    v = np.copy(v1)

    dot_v1_v2 = np.dot(v1, v2)
    norm_v2 = np.linalg.norm(v2)

    if abs(dot_v1_v2) > norm_v2 ** 2 / 2:  # corrected norm_v2 / 2 -> norm_v2 ** 2 / 2 by V. Baturin 09.11.18
        try:
            v1_trial = v1 - np.ceil(abs(dot_v1_v2) / (np.linalg.norm(v2) ** 2)) * np.sign(dot_v1_v2) * v2
        except:
            v1_trial = v1

        if np.linalg.norm(v1_trial) < np.linalg.norm(v1):
            v = v1_trial
            flag = 1

    return v, flag


def optLattice(coor, lattice):
    """
    Idea is as follows - if any lattice vector has projection onto any other lattice vector
    that is greater than half of length of this vector, we can reoptimize the shape
    i. e. if |a*b|/|b| > |b|/2 then a_new = a - ceil(|a*b|/|b|^2)*sign(a*b)*b

    :type coor: list or numpy array
    :param coor: Nx3 coordinates.
    :type lattice: list or numpy array
    :param lattice: 3x3 representation of the lattice.
    :rtype: tuple
    :return: (optcoor, optlat) with optimized coordinates (Nx3 NumPy array) and optimized lattice (3x3 NumPy array).
    """

    coor = np.array(coor)
    lattice = np.array(lattice)

    v1 = lattice[0, :]
    v2 = lattice[1, :]
    v3 = lattice[2, :]

    flag = 1
    step = 0

    while flag:
        flag = 0
        v1, flag = reoptimizeVector(v1, v2, flag)
        v1, flag = reoptimizeVector(v1, v3, flag)
        v2, flag = reoptimizeVector(v2, v1, flag)
        v2, flag = reoptimizeVector(v2, v3, flag)
        v3, flag = reoptimizeVector(v3, v1, flag)
        v3, flag = reoptimizeVector(v3, v2, flag)

        if step > 100:
            break  # so that the cycle isn't infinite

        v1, flag = reoptimizeVector(v1, v2 + v3, flag)
        v2, flag = reoptimizeVector(v2, v1 + v3, flag)
        v3, flag = reoptimizeVector(v3, v1 + v2, flag)

        if flag:
            step += 1

    optlat = np.zeros((3, 3))

    optlat[0, :] = v1
    optlat[1, :] = v2
    optlat[2, :] = v3

    if np.linalg.det(optlat) < 0.0000001 and np.sum(coor) == 0:
        tempcoor = np.copy(coor)
        optlat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        tempcoor = np.dot(coor, np.linalg.inv(optlat))

    optcoor = tempcoor - np.floor(tempcoor)  # now coordinates are fractional
    optcoor = np.dot(optcoor, optlat)  # now absolute again for molecules

    return optcoor, optlat
