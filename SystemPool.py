"""
USPEX.Common.SystemPool
=======================

Contains configuration of such space, parameters of what we are searching for

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from typing import List, Tuple
import numpy as np

from .paretoRanking import paretoRanking
from .Fitness import Fintness

import logging
logger = logging.getLogger(__name__)


def _fitnessRepresentation(fitness: List[Tuple[str, str]]):
    return '_'.join(f'{attr}_{direction}' for (attr, direction) in sorted(fitness, key=lambda entry: entry[0]))


class SystemPool(object):
    """
    This class contains configuration of such space, parameters of what are we searching for,
    list of systems already studied in the search, current result of the search.

    This class is universal for all kind of targets and should be inherited for particular targets
    and its methods overridden.

    :cvar DEFAULT_FITNESS:
        list of tuples describing default optimization type for this pool of systems.
        Each tuple consist of name of attribute and 'min' or 'max' modifier. Class variable.
        For default implementation this list is empty.
    :ivar config:
        link to implementation of :class:`~USPEX.Common.Config.Config` interface.
    :ivar best:
        list of currently best known systems.
    :ivar uniqueSystems:
        list of all currently studied systems.
    """

    DEFAULT_FITNESS = []

    def __init__(self):
        """
        Initializes the class.

        :type config: :class:`~USPEX.Common.Config.Config` or descendant
        :param config: describes the chemical compositions configuration space.
        """
        self.best = {}
        self.uniqueSystems = []
        self._newID = 0
        self.fitness = Fintness()

    def update(self, population: list):
        """
        Update information about target space in current search.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """

        self.best = {}

        logger.info('Updating target: list of unique systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('add new system %d to list of unique systems' % system.ID)
                uniqueIDs.append(system.ID)
                self.uniqueSystems.append(system)

    def newFoundSystems(self, population: list):
        """
        Determines new found systems in population.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        :rtype: list
        :return: new found systems.
        """
        logger.info('Determine new systems.')
        uniqueIDs = [system.ID for system in self.uniqueSystems]
        newFoundSystems = []
        for system in population:
            if system.ID not in uniqueIDs:
                logger.debug('found new system %d' % system.ID)
                newFoundSystems.append(system)
                uniqueIDs.append(system.ID)
        return newFoundSystems

    def cleanDuplicates(self, population: list):
        """
        Method for cleaning duplicates. It usually needs specific redefinition in the respective child classes.

        :type population: list of :class:`~USPEX.Common.System.System` descendants
        :param population: list of systems which allows to update our knowledge about target space.
        """
        pass

    def setBest(self, fitness: List[Tuple[str, str]], best: list):
        """
        Sets the best individuals as an attribute of the class.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        :type best: list
        :param best: best individuals according to the fitness.
        """
        self.best[_fitnessRepresentation(fitness)] = best

    def getBest(self, fitness: List[Tuple[str, str]]):
        """
        Gets the best individuals according to the fitness.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        """
        return self.best[_fitnessRepresentation(fitness)]

    def sort(self, fitness: List[Tuple[str, str]], population: list):
        """
        Method for sorting our population by fitness.

        :type fitness: list[tuple[str]]
        :param fitness: list of tuples ('property', 'direction'), where direction is 'min' or 'max'.
        :type population: list
        :param population: unsorted list of structures.
        :rtype: list
        :return: sorted population.
        """
        return self.fitness.sort(fitness, population, self.uniqueSystems)

    def assignID(self, system):
        """
        Assign ID to system.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: system to be labeled with ID.
        """
        system.ID = self._newID
        self._newID += 1
