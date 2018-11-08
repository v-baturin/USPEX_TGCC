from __future__ import division

import numpy as np


# TODO check this functions and write tests for them

def reoptimizeVector(v1, v2, flag1):
    """
    The function reoptimizes a vector. Need better description.
    :param v1:
    :param v2:
    :param flag1:
    :return v:
    :return flag:
    """

    flag = flag1

    # Convert vectors to ndarray:
    v1 = np.array(v1)
    v2 = np.array(v2)

    v = np.copy(v1)

    dot_v1_v2 = np.dot(v1, v2)
    norm_v2 = np.linalg.norm(v2)

    if abs(dot_v1_v2) > norm_v2 / 2: # Seems erroneous
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

    :param coor: 1x3 list (or can be 1x3 NumPy array).
    :param lattice: 3x3 representation of the lattice (either Python list or NumPy array).
    :return: optcoor: optimized coordinates (1x3 NumPy array).
    :return: optlat: optimized lattice (3x3 NumPy array).
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


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    '''
    coord = [0, 0, 0]  # Fictitious coordinates
    lattice = [[8.8961357250244362, 0.0, 0.0], [7.4920051751263292, 5.8541498910494623, 0.0],
               [1.4497839312305156, 0.72741049512938449, 4.172098127196039]]

    lattice = np.asarray(lattice)
    v1 = lattice[0, :]
    v2 = lattice[1, :]
    v, flag = reoptimizeVector(v1, v2, 0)
    '''

    from lib.mat2dict import loadmat

    test_dir = 'test_optLattice'

    coor = loadmat(test_dir + '/coor.mat')['coor']
    lattice = loadmat(test_dir + '/lattice.mat')['lattice']

    optcoor, optlat = optLattice(coor, lattice)

    print('')
