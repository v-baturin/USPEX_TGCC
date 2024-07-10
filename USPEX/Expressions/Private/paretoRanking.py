"""
USPEX.Common.paretoRanking
==========================

Function for detecting Pareto fronts

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np


def paretoRanking(fitnesses) -> list[list[int]]:
    """
    Function for detecting Pareto fronts.

    :type fitnesses: numpy array
    :param fitnesses: 2xN array with the value of each fitness for each structure.
    """
    fitnesses = np.asarray(fitnesses, dtype=float)
    size, N = fitnesses.shape  # number of fitnesses
    fronts = []
    unranked = np.arange(size)

    while len(unranked) > 0:
        A = np.all(fitnesses[unranked].reshape((len(unranked), 1, N)) >=
                   fitnesses[unranked].reshape((1, len(unranked), N)), axis=2)
        B = np.logical_not(np.all(fitnesses[unranked].reshape((len(unranked), 1, N)) ==
                                  fitnesses[unranked].reshape((1, len(unranked), N)), axis=2))
        front = unranked[np.flatnonzero(np.logical_not(np.any(np.logical_and(A, B), axis=1)))]
        fronts.append(front.tolist())
        unranked = np.asarray(list(set(unranked) - set(front)))

    return fronts
