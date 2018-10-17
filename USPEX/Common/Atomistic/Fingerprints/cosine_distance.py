__author__ = 'mrakitin'

import numpy as np


def cosine_distance(matrA, matrB, weight):
    """
    Calculation of cosine distances using eq.(6b) from JCP-2009.
    :param matrA: atomic fingerprint 1.
    :param matrB: atomic fingerprint 2.
    :param weight: weight for a particular atom type in the cell.
    :return dist: resulted distance.
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


if __name__ == "__main__":
    from lib.mat2dict import loadmat

    test_dir = 'test_cosine_distance'

    # Calc type 301:
    matrA = loadmat(test_dir + '/matrA.mat')['matrA']
    matrB = loadmat(test_dir + '/matrB.mat')['matrB']
    # In Python we have (1x125) arrays:
    matrA = matrA.reshape((1, matrA.shape[0]))
    matrB = matrB.reshape((1, matrB.shape[0]))
    weight = np.asarray([1])

    # Calc type 300:
    '''
    matrA = loadmat(test_dir + '/matrA_300.mat')['matrA']
    matrB = loadmat(test_dir + '/matrB_300.mat')['matrB']
    weight = loadmat(test_dir + '/weight_300.mat')['weight']
    weight = np.diag(weight)
    '''

    dist = cosine_distance(matrA, matrB, weight)

    print('')
