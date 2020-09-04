"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from typing import List, Tuple

from .Target import Target

logger = logging.getLogger(__name__)


class GlobalOptimizer(object):
    """
    Main purpose of this class is to generate new structures
    that will be then optimized and selected best of them to the output.
    """

    Fitness = None
    knownSelectionTypes = {}

    def __init__(self, target: dict, selection: dict, fitness, stopFitness=None, output=None, **kwargs):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        :type output: :class:`~USPEX.Common.Output.Output`
        :param output: instance of class handling output.
        """

        self.target = Target(**target)

        assert self.Fitness is not None
        self.fitness = self.Fitness(self.target.pool, self.target.utilities)
        self.fitnessConvergence = fitness
        self.best = set()
        self._isStable = False
        self.stopFitness = stopFitness
        self._isGoalReached = False

        self.selectionConfig = selection
        self.createPopulation = self.knownSelectionTypes[selection['type']](**selection)

        # List of structure recieved from update on this particular step
        self.population = None
        # List of new found structure on this particular step
        self.newStructures = None

        self.output = output

    def run(self):
        """
        Here we generate new set of structures.

        :rtype: list
        :return: list of structures.
        """
        return self.createPopulation(self.target, self.fitness, self.population, self.newStructures)

    def update(self, population: list):
        """
        Updates state of optimized structures and write current state of them into the output.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        self.target.pool.cleanDuplicates(population)
        self.population = population
        self.newStructures = self.target.pool.newFoundSystems(population)
        self.target.pool.update(self.newStructures)
        best = set(system.ID for system in self.fitness.sort(self.fitnessConvergence, self.target.pool.uniqueSystems)[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best
        if self.stopFitness is not None:
            for ID in list(self.best):
                value = self.fitness.getFitnessByID(self.fitnessConvergence, ID)
                if value is None:
                    try:
                        value = getattr(self.target.pool.allSystems[ID], self.fitnessConvergence)
                    except:
                        pass
                if round(value, ndigits=3) <= round(self.stopFitness, ndigits=3):
                    self._isGoalReached = True

    @property
    def isStable(self):
        return self._isStable

    @property
    def isGoalReached(self):
        return self._isGoalReached

    @classmethod
    def setFitnessType(cls, FitnessType: type):
        cls.Fitness = FitnessType

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes
        cls.knownSelectionTypes[selectionType.__name__] = selectionType
