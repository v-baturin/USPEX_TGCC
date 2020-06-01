"""
USPEX.Common.SystemPool
=======================

Contains configuration of such space, parameters of what we are searching for

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

from typing import List, Tuple
import numpy as np

from .paretoRanking import paretoRanking

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

    def antiseedsCorrection(self, system) -> float:
        return 0

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
        populationFitnesses = []

        for system in population:
            systemFitnesses = []
            for attribute, direction in fitness:
                if direction == 'min':
                    factor = 1
                    correction = 0
                elif direction == 'max':
                    factor = -1
                    correction = 0
                elif direction == 'min_antiseeds':
                    factor = 1
                    correction = self.antiseedsCorrection(system)
                elif direction == 'max_antiseeds':
                    factor = -1
                    correction = self.antiseedsCorrection(system)
                else:
                    factor = 1
                    correction = 0
                    logger.debug('Incorrect optimization deirection "{}" using default "min"'.format(direction))
                if hasattr(self, attribute):
                    value = factor * getattr(self, attribute)(system)
                elif hasattr(system, attribute):
                    value = factor * getattr(system, attribute)
                else:
                    value = 0
                    logger.debug('Neither target {} nor system {} has attribute {}'
                                 ' which is set as fitness.'.format(self.__name__, system.ID, attribute))
                systemFitnesses.append((value, correction))
            populationFitnesses.append(systemFitnesses)

        logger.debug('Fitnesses of this population are: {}'.format(populationFitnesses))
        populationFitnesses = np.asarray(populationFitnesses)
        populationFitnessesValues = populationFitnesses[:,:,0]
        populationFitnessesCorrections = populationFitnesses[:, :, 1]
        populationFitnessesValues += (populationFitnessesValues.mean(axis = 0) -
                                      populationFitnessesValues.min(axis = 0)).reshape((1,-1)) * populationFitnessesCorrections
        ranking = paretoRanking(populationFitnessesValues.tolist())
        return [[population[index] for index in front] for front in ranking]
        # uniqueFinesses, ranking = np.unique(populationFitnesses, return_inverse=True)
        # return [[population[ind] for ind in (ranking == rank).nonzero()[0]] for rank in range(len(uniqueFinesses))]

    def assignID(self, system):
        """
        Assign ID to system.

        :type system: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` descendant
        :param system: system to be labeled with ID.
        """
        system.ID = self._newID
        self._newID += 1
