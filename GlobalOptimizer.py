"""
USPEX.Common.GlobalOptimizer
============================

Class implementing global optimizer

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
from .Worker import Worker
from .Target import Target
from .Selection import Selection


logger = logging.getLogger(__name__)


class GlobalOptimizer(Worker):
    """
    Main purpose of this class is to generate new structures
    that will be then optimized and selected best of them to the output.
    """

    def __init__(self, target: dict, selection: dict, newoutput=None):
        """
        Initializes the class.

        :type target: dict{type, params}
        :param target: name of target system and its parameters; obligatory
        :type selection: dict{type, params}
        :param selection: name of selection to launch and its parameters; obligatory
        :type newoutput: :class:`~USPEX.Common.Output.Output`
        :param newoutput: instance of class handling output.
        """

        self.target = Target(**target)
        if 'fitness' in target:
            self.fitness = target['fitness']
        else:
            self.fitness = self.target.pool.DEFAULT_FITNESS

        self.selection = Selection(self.target, **selection)
        self.population = None
        # List of new found structure on this particular step
        self.newStructures = None

        self.output = newoutput
        if self.output is not None:
            self.output.targetConfig = self.target.config
            self.output.selectionConfig = self.selection.config

    def run(self, population: list = None):
        """
        Here we generate new set of structures.

        :rtype: list
        :return: list of structures.
        """

        if population is None and self.population is not None:
            return self.population

        self.population, *analysis = self.selection.createPopulation(population, self.newStructures, self.fitness)

        self.save()

        if self.output is not None:
            self.output.handleAnalysis(analysis)

        return self.population

    def update(self, population: list):
        """
        Updates state of optimized structures and write current state of them into the output.

        :type population: list
        :param population: list of systems which allows to update our knowledge about target space.
        """
        self.target.pool.cleanDuplicates(population)
        self.newStructures = self.target.pool.newFoundSystems(population)
        self.target.pool.update(self.newStructures)
        best = self.target.pool.sort(self.fitness, self.target.pool.uniqueSystems)[0]
        self.target.pool.setBest(self.fitness, best)

        if self.output is not None:
            self.output.handlePool(self.target.pool)
