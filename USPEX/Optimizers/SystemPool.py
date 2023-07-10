"""
USPEX.SystemPool
=======================
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


import logging
import numpy as np
from copy import copy

from USPEX.Expressions.Functions.presets import applyPresetsRecursive

logger = logging.getLogger(__name__)


class SystemPool(object):
    """
    This class serves as database of all systems encountered in calculation.
    The only thing it expects from systems is IDs. It has a method to assign IDs to systems.
    Except for that systems are arbitrary dictionaries.
    It subdivide systems into generations. Each call to *update* method creates new generation record.
    """

    entryFactory = None

    def __init__(self):
        self.extensions = {}
        self.allSystems = {}
        self.generations = []
        self.goodSystemIDs = []
        self._newID = 0

    def __copy__(self):
        other = SystemPool.__new__(SystemPool)
        other.allSystems = copy(self.allSystems)
        other.generations = copy(self.generations)
        other._newID = self._newID
        return other

    @property
    def goodSystems(self):
        return tuple(self.allSystems[ID] for ID in self.goodSystemIDs)

    @property
    def uniqueSystems(self):
        return tuple(self.allSystems[ID] for ID in self.goodSystemIDs if self.allSystems[ID].originalID is None)

    @property
    def uniqueSystemIDs(self):
        """
        :return: list of IDs of unique structures.
        """
        return tuple(ID for ID in self.goodSystemIDs if self.allSystems[ID].originalID is None)

    def __hash__(self):
        return hash(self.uniqueSystemIDs)

    def update(self, population: list):
        """
        Update information about target space in current search.

        :param population: list of structures.

        """

        for system in population:
            self.allSystems[system['ID']] = system

    def append(self, population):
        """
        """
        logger.debug('Updating target: list of unique systems.')
        IDs = set(system['ID'] for system in population)
        newIDs = []
        newGeneration = {'allSystems': [], 'newSystems': []}
        for system in population:
            original = self.allSystems[self.getOriginalID(system['ID'])]
            if original['ID'] not in newIDs:
                newGeneration['allSystems'].append(original)
                newIDs.append(original['ID'])
                if set(original.duplicates) <= IDs:
                    logger.debug(f'add new system {system["ID"]} to list of unique systems')
                    newGeneration['newSystems'].append(original)
        self.generations.append(newGeneration)

    def assignID(self, system):
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        system.ID = self._newID
        system.setProperty('isBad', True)
        self._newID += 1
        self.allSystems[system.ID] = system

    def getOriginalID(self, ID):
        """
        If system is duplicate return ID of original system otherwise return input ID.

        :param ID: ID of some system from this pool.

        :return: ID of original system.
        """
        system = self.allSystems[ID]
        return system.originalID if system.originalID is not None else ID

    @staticmethod
    def fronts(pool, expression):
        expression = applyPresetsRecursive(expression)
        values = [s[expression] for s in pool]
        return [[pool[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

