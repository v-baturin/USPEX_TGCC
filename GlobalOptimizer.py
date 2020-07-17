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

    def __init__(self, target: dict, selection: dict, fitness: List[Tuple[str, str]], output=None, **kwargs):
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
        self.fitness = self.Fitness()
        self.fitnessConvergence = fitness
        self.best = set()
        self._isStable = False

        self.createPopulation = self.knownSelectionTypes[selection['type']](**selection)

        # List of new found structure on this particular step
        self.newStructures = None

        self.output = output
        if self.output is not None:
            self.output.targetConfig = self.target.config
            self.output.selectionConfig = selection

    def run(self, population: list = None):
        """
        Here we generate new set of structures.

        :rtype: list
        :return: list of structures.
        """

        population, *analysis = self.createPopulation(self.target, self.fitness, population, self.newStructures)

        if self.output is not None:
            self.output.handleAnalysis(analysis)

        return population

    def update(self, population: list):
        """
        Updates state of optimized structures and write current state of them into the output.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        self.target.pool.cleanDuplicates(population)
        self.newStructures = self.target.pool.newFoundSystems(population)
        self.target.pool.update(self.newStructures)
        best = set(system.ID for system in self.fitness.sort(self.fitnessConvergence, self.target.pool.uniqueSystems,
                                                             self.target.pool.uniqueSystems)[0])
        if best == self.best:
            self._isStable = True
        else:
            self._isStable = False
            self.best = best

        if self.output is not None:
            self.output.handlePool(self.target.pool)

    def isStable(self):
        return self._isStable

    def isGoalReached(self):
        return False

    @classmethod
    def setFitnessType(cls, FitnessType: type):
        cls.Fitness = FitnessType

    @classmethod
    def registerSelection(cls, selectionType: type):
        assert selectionType.__name__ not in cls.knownSelectionTypes
        cls.knownSelectionTypes[selectionType.__name__] = selectionType
