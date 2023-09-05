"""
USPEX.SystemPool
=======================
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


import logging
import numpy as np
from copy import copy, deepcopy

from USPEX.Expressions.Functions.presets import applyPresetsRecursive
from .PoolEntry import PoolEntry, EntryFlavour

logger = logging.getLogger(__name__)


class SystemPool(object):
    """
    This class serves as database of all systems encountered in calculation.
    The only thing it expects from systems is IDs. It has a method to assign IDs to systems.
    Except for that systems are arbitrary dictionaries.
    It subdivide systems into generations. Each call to *update* method creates new generation record.
    """


    def __init__(self, flavourFactory):
        self.allSystems = {}
        self.generations = []
        self.goodSystemIDs = []
        self._newID = 0
        self.flavourFactory = flavourFactory

    def __copy__(self):
        other = SystemPool(self.flavourFactory)
        other.allSystems = copy(self.allSystems)
        other.generations = deepcopy(self.generations)
        other.goodSystemIDs = copy(self.goodSystemIDs)
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

    def newEntry(self, system: EntryFlavour) -> PoolEntry:
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        entry = PoolEntry(self._newID, system)
        self._newID += 1
        self.allSystems[entry.ID] = entry
        logger.info(f"System {entry.ID} successfully created by {entry['.howCome.origin']} operator"
                    f" from {entry['.parent.origin']} parents.")
        return entry

    @staticmethod
    def fronts(pool, expression):
        expression = applyPresetsRecursive(expression)
        values = [s[expression] for s in pool]
        return [[pool[ind] for ind in np.flatnonzero(values == value)] for value in np.unique(values)]

