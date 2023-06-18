"""
USPEX.Common.Fitness
====================

Data type representing rules of how we determine which systems are better.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from copy import copy
from sklearn.decomposition import PCA

from .Private.ConvexHull import ConvexHull
from .Private.paretoRanking import paretoRanking


logger = logging.getLogger(__name__)


class BasicFunctions:

    @staticmethod
    def min(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.min(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def max(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.max(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def mean(values: np.ndarray) -> np.ndarray:
        return np.expand_dims(np.mean(values, axis=0),axis = 0).repeat(len(values), axis=0)

    @staticmethod
    def plus(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 + values2

    @staticmethod
    def minus(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 - values2

    @staticmethod
    def multiply(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 * values2

    @staticmethod
    def divide(values1: np.ndarray, values2: np.ndarray) -> np.ndarray:
        return values1 / values2

    @staticmethod
    def negate(values: np.ndarray) -> np.ndarray:
        return values * (-1)

    @staticmethod
    def tabulate(systems: np.ndarray) -> np.ndarray:
        keys = set()
        for system in systems:
            assert isinstance(system, Mapping), type(system)
            keys.update(system.keys())
        keys = sorted(keys)
        table = []
        for system in systems:
            row = []
            for key in keys:
                # do not change to *if key in system*
                try:
                    row.append(system[key])
                except KeyError:
                    row.append(0)
            table.append(row)
        return np.asarray(table)

    @staticmethod
    def hstack(table: np.ndarray) -> np.ndarray:
        return np.hstack(table.transpose((1,0,2)))

    @staticmethod
    def getPrincipalComponents(dimensionality: int, data: np.ndarray) -> np.ndarray:
        principalComponents = PCA(n_components=dimensionality).fit_transform(data)
        assert principalComponents.shape[0] == data.shape[0]
        return principalComponents

    @staticmethod
    def convexHullHeight(space: np.ndarray) -> np.ndarray:
        return copy(ConvexHull(space).height)

    @staticmethod
    def simpleHeight(space: np.ndarray) -> np.ndarray:
        nRows, nColumns = space.shape
        if nColumns > 1:
            values = np.empty(nRows, dtype=float)
            for row in space:
                arg = np.empty((nColumns - 1, nRows), dtype=float)
                *arg[:], value = (space - row).T
                where = np.isclose(arg.all(axis=0), 0).nonzero()
                value = value[where]
                values[where] = value - value.min()
        else:
            values = (space - space.min(axis=0)).flatten()
        return values

    @staticmethod
    def getRelativeCHSpace(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        assert arguments.shape[0] == properties.shape[0]
        arguments = arguments.astype(float)
        totalBlocks = arguments.sum(axis=1)
        arguments /= totalBlocks.reshape((-1, 1))
        properties = np.nan_to_num(properties) / totalBlocks
        arguments[:,-1] = properties
        return arguments

    @staticmethod
    def getAbsoluteCHSpace(arguments: np.ndarray, properties: np.ndarray) -> np.ndarray:
        poolSize, argNum = arguments.shape
        assert properties.shape == (poolSize,)
        space = np.empty((poolSize, argNum + 1), dtype=float)
        space[:, :-1] = arguments
        space[:, -1] = np.nan_to_num(properties)
        return space

    @staticmethod
    def pareto(*arguments) -> np.ndarray:
        arguments = np.nan_to_num(np.asarray(arguments)).T.tolist()
        values = np.empty((len(arguments),), dtype=int)
        for i, front in enumerate(paretoRanking(arguments)):
            for ind in front:
                values[ind] = i
        return values
