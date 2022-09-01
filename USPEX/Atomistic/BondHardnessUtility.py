"""
USPEX.Atomistic.BondHardnessUtility
===================================
"""

import logging
import numpy as np
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

closest = np.array(
    [[[0, 0, 0]], [[-1, 0, 0]], [[-1, 0, -1]], [[-1, -1, -1]], [[-1, -1, 0]], [[0, -1, 0]], [[0, -1, -1]],
     [[0, 0, -1]], [[-1, -1, 1]], [[-1, 0, 1]], [[-1, 1, 1]], [[-1, 1, 0]], [[-1, 1, -1]], [[0, -1, 1]],
     [[0, 0, 1]], [[0, 1, 1]], [[0, 1, 0]], [[0, 1, -1]], [[1, 0, 0]], [[1, 0, -1]], [[1, -1, -1]],
     [[1, -1, 0]], [[1, -1, 1]], [[1, 0, 1]], [[1, 1, 1]], [[1, 1, 0]], [[1, 1, -1]]])


class BondHardnessUtility:
    def __init__(self):
        pass

    def calcBondHardness(self, structure):
        pass

    @staticmethod
    def calcCoordinationNumbers(structure):
        radii = np.tile([element.covalent_radius for element in structure.getAtomTypes()], len(closest))
        base = radii.reshape(1, radii.size) + radii.reshape(radii.size, 1)
        vertices = np.dot(np.concatenate((closest + structure.getFractionalCoordinates()), axis=0),
                          structure.getCell().getCellVectors())
        order = np.exp(-(cdist(vertices, vertices) - base) / 0.23)
        order = np.delete(np.triu(order, 1), 0, 1) + np.delete(np.tril(order, -1), len(vertices) - 1, 1)
        return order.sum(axis=1) / order.max(axis=1)
