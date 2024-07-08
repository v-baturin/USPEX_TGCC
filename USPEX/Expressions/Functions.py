"""
USPEX.Common.Fitness
====================

Data type representing rules of how we determine which systems are better.

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from typing import Mapping
from copy import copy
from sklearn.decomposition import PCA

from USPEX.Expressions.Private.ConvexHull import ConvexHull
from USPEX.Expressions.Private.paretoRanking import paretoRanking


logger = logging.getLogger(__name__)


class Functions:

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

    @staticmethod
    def applyCorrections(values: np.ndarray, corrections: np.ndarray) -> np.ndarray:
        return (values.mean() - values.min())*corrections + values

presetFitness = {
    ('aging', 'values'): ('applyCorrections', 'values', 'antiseeds.corrections.origin'),
    ('getRelCCHSpace', 'values'): ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition.origin'), 'values'),
    ('getAbsCCHSpace', 'values'): ('getAbsoluteCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition.origin'), 'values'),
    ('heightCCH', 'values'): ('convexHullHeight', ('getRelCCHSpace', 'values')),
    ('heightCS', 'values'): ('simpleHeight', ('getRelCCHSpace', 'values')),
    # 'energyCCH': ('heightCCH', 'energy'),
    # 'enthalpyCCH': ('heightCCH', 'enthalpy'),
    # 'energyCS': ('heightCS', 'energy'),
    # 'enthalpyCS': ('heightCS', 'enthalpy'),
    # 'refinedEnergy': ('divide', ('minus', 'energy', 'onlyEnvironment.energy'), 'supercellFactor'),
    # 'refinedEnthalpy': ('divide', ('minus', 'enthalpy', 'onlyEnvironment.enthalpy'), 'supercellFactor'),
    # 'refinedEnergyCCH': ('heightCCH', 'refinedEnergy'),
    # 'refinedEnthalpyCCH': ('heightCCH', 'refinedEnthalpy'),
    # 'normRefinedEnthalpy': ('divide', 'refinedEnthalpy', 'cellUtility.area'),
    # 'normRefinedAbsCompCH': ('convexHullHeight', ('getAbsCCHSpace', 'normRefinedEnthalpy')),
    # 'refinedInterfaceEnergy': ('minus', ('minus', 'energy', 'onlyLowerEnvironment.energy'),
    #                            'onlyUpperEnvironment.energy'),
    # 'refinedInterfaceEnergyCCH': ('heightCCH', 'refinedInterfaceEnergy'),
}

def applyPresets(optType):
    optType_ref = optType
    if optType in presetFitness:
        optType = presetFitness[optType]
    elif isinstance(optType, tuple):
        funcName, *funcParams = optType
        for probeFitness in presetFitness.keys():
            if isinstance(probeFitness, tuple) and probeFitness[0] == funcName and len(probeFitness) == len(optType):
                funcName, *templateParams = probeFitness
                optType = presetFitness[probeFitness]
                for param, templateParam in zip(funcParams, templateParams):
                    optType = _substituteParams(optType, templateParam, param)
                break
    if optType != optType_ref:
        optType = applyPresets(optType)
    return optType

def applyPresetsRecursive(optType):
    if isinstance(optType, list):
        optType = tuple(optType)
    optType = applyPresets(optType)
    if isinstance(optType, tuple):
        optType = (optType[0], *(applyPresetsRecursive(param) for param in optType[1:]))
    return optType


def _substituteParams(optType, templateParam, param):
    if optType == templateParam:
        optType = param
    elif isinstance(optType, tuple):
        funcName, *funcParams = optType
        optType = (funcName,)
        for funcParam in funcParams:
            optType += (_substituteParams(funcParam, templateParam, param),)
    return optType

