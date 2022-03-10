"""
USPEX.SystemPool
=======================
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""


import logging
from copy import copy
from itertools import chain

logger = logging.getLogger(__name__)


class SystemPool(object):
    """
    This class serves as database of all systems encountered in calculation.
    The only thing it expects from systems is IDs. It has a method to assign IDs to systems.
    Except for that systems are arbitrary dictionaries.
    It subdivide systems into generations. Each call to *update* method creates new generation record.
    """

    def __init__(self):
        self.uniqueSystems = ()
        self.allSystems = {}
        self.generations = []
        self._newID = 0

    def __copy__(self):
        other = SystemPool.__new__(SystemPool)
        other.uniqueSystems = self.uniqueSystems
        other.allSystems = copy(self.allSystems)
        other.generations = copy(self.generations)
        other._newID = self._newID
        return other

    def getUniqueIDs(self):
        """
        :return: list of IDs of unique structures.
        """
        return [system['ID'] for system in self.uniqueSystems]

    def __hash__(self):
        return hash(tuple(self.getUniqueIDs()))

    def update(self, population: list):
        """
        Update information about target space in current search.

        :param population: list of structures.

        """

        logger.debug('Updating target: list of unique systems.')
        uniqueIDs = self.getUniqueIDs()
        newGeneration = {'allSystems': [], 'newSystems': []}
        for system in population:
            newGeneration['allSystems'].append(system)
            if system['ID'] not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system['ID'])
                uniqueIDs.append(system['ID'])
                newGeneration['newSystems'].append(system)
        self.generations.append(newGeneration)
        self.uniqueSystems = tuple(chain.from_iterable(generation['newSystems'] for generation in self.generations))

    def updateFitness(self, fitness):
        """
        Inserts fitness object into last generation record.

        :param fitness: fitness object.

        """
        assert 'fitness' not in self.generations[-1]
        self.generations[-1]['fitness'] = fitness

    def assignID(self, system):
        """
        Assign ID to system.

        :type system:
        :param system: system to be labeled with ID.

        """
        system['ID'] = self._newID
        self._newID += 1
        self.allSystems[system['ID']] = system

    def getOriginalID(self, ID):
        """
        If system is duplicate return ID of original system otherwise return input ID.

        :param ID: ID of some system from this pool.

        :return: ID of original system.
        """
        system = self.allSystems[ID]
        return system['originalID'] if 'originalID' in system else ID